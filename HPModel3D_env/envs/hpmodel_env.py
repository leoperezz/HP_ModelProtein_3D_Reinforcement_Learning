import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pickle

POLY_TO_INT = {
    'H':1,'P':-1
}

ACTION_TO_STR = {
    0:'X0',
    1:'X1',
    2:'Y0',
    3:'Y1',
    4:'Z0',
    5:'Z1',
    6:'NULL'
}

#(Z,Y,X)
MOVS = np.array([
    [0, 0,-1], #X0 
    [0, 0, 1], #X1 
    [0,-1, 0], #Y0 
    [0, 1, 0], #Y1 
    [-1,0, 0], #Z0 
    [ 1,0, 0],  #Z1
    [0 ,0, 0]  #null 
])

N_ACTIONS = 7


class Node:

    """
    Node:
    -----
    It is a class that contains the attributes of polymers 
    such as position, type, and order.    

    """

    def __init__(self,id: int ,type: str ,pos_array: np.array):

        """
        Parameters:
        -----------

        (int) id :

        Represents the position in the sequence

        (str) type:

        Represents the type of polymer e.g H or P

        (np.array) pos_array:

        Represents the position with coordinates (z,y,x)


        """

        self.id = id
        self.type = type
        self.pos_array = pos_array

    def _calculate_adjacents(self,node):

        """
        Parameters:
        ----------

        (Node) node: 
        
        The node for which we want to know if it is adjacent or not
        
        Returns:
        --------

        True if is adjacent or False otherwise

        """

        distance = np.linalg.norm(self.pos_array-node.pos_array)

        if distance == 1.0:
            return True
        else:
            return False



            



def convert_one_hot_seq(seq : str):

    """
    Parameters:
    ----------

    (str) seq:

    The sequence of a polymer e.g HPPPHPPHPH

    Returns:
    -------
    Its representation of one_hot with shape of (n,2) 
    where n is the length of the sequence. 

    """

    r_seq = torch.tensor([0 if i == 'H' else 1 for i in seq])
    
    one_hot = F.one_hot(r_seq, num_classes = 2)

    return one_hot



class HPModel3D(gym.Env):


    def __init__(self,seq):
        """
        Parameters
        ----------
        (str)seq:
        
        Sequence contains the polymer chain.
        Must only consist of 'H' or 'P'.

        """        
        super(HPModel3D,self).__init__()

        self.seq = seq

        self.seq_one_hot = convert_one_hot_seq(seq)#Sequence in one_hot

        n = len(seq) #the size of the sequence

        l = 2*n + 1  #dimension of the observation_space

        self.shape_grid = (n,l,l)

        self.midpoint  = (0,l//2,l//2)

        self.action_space = spaces.Discrete(6)  
        
        self.observation_space = spaces.Box(
            low=-1.0,high=1.0,shape=self.shape_grid, dtype = int
        )

        self.reset() #initialize all values



    def step(self, action : int ):

        """
        Parameters:
        -----------

        (int) action: 

        It's the action to execute in the environment

        Returns:
        -------

        (torch.tensor) next_state:

        A tensor containing the one_hot representation of the sequence
        concatenated to the actions performed up to that point also in 
        its one_hot representation.

        (float) reward:

        The reward of the enviroment given an action.

        (int) done:

        Returns 1 if the sequence is complete or any action can't be performed,
        otherwise 0

        (dict) info:

        A dictionary containing the information for the last polymer placed, 
        as well as the actions performed up to that point.

        """

        n,l,l = self.shape_grid

        new_pos = self.state['pos_array'] + MOVS[action,:]

        z,y,x =  new_pos

        #1)Verify if the action is possible or not:

        #1.1) Verify if exist any posible solution:

        exist_solution = self._verify_solution()

        reward = self._calculate_reward()-(len(self.seq)-len(self.buffer_nodes)) 
        done = 1 #if not exist, the episode is over
        info = self.state


        if not exist_solution:

            return self._preprocess_state(self.state),reward,done,info


        #1.2) Verify if the new_pos is valid or not:
        
        done = 0
        info = self.state


        valid = (z>=0) & (z<n) & (y>=0) & (y<l) & (x>=0) & (x<l)

        reward = -2 #Reward of invalid pos

        if not valid:

            return self._preprocess_state(self.state),reward,done,info


        #1.3) Verify if some polymer is in the new_pos

        reward = -2 #Reward of colission

        if self.grid[z,y,x]!=0:

            return self._preprocess_state(self.state),reward,done,info
    

        #1.4) Is a valid move

        #verify if is H or not
        is_H = True if self.seq[self.state['id']+1] == 'H' else False 

        #Update the state of the enviroment

        actions = self.state['actions']

        actions[self.state['id']+1] = action

        self.state = {
            'id':self.state['id']+1,
            'pos_tuple':(z,y,x),
            'pos_array': new_pos,
            'is_H':is_H,
            'actions':actions
        }

        #Update the grid
        if is_H : 
            self.grid[z,y,x] = 1 
        else: 
            self.grid[z,y,x] = -1

        node = Node(self.state['id'],self.seq[self.state['id']],self.state['pos_array'])

        self._update_buffer_relations(node) #update the relations in the buffer

        self.buffer_nodes.append(node)    

        done = 1 if self.state['id']+1 == n else 0

        if done:
            reward = self._calculate_reward()
        else:
            reward = 0 #reward of a normal action

        info = self.state        

        return  self._preprocess_state(self.state),reward,done,info



    def reset(self):

        #contains the relations between H polymers e.g (1.2) represent 
        #the relation between H1 and H2 polymer. Further, if the set contains
        #(x,y) it will contain (y,x) too.       
        self.buffer_relations = set()

        self.buffer_nodes = list()

        self.grid = np.zeros(self.shape_grid) #initialize the grid

        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]#start in the midpoint

        is_H = True if self.seq[0] == 'H' else False 

        self.state = {
            'id':0,
            'pos_tuple': self.midpoint,
            'pos_array': np.array(self.midpoint),
            'is_H': is_H,
            'actions': [N_ACTIONS-1]*len(self.seq)
        }

        node = Node(0,self.seq[self.state['id']],self.state['pos_array'])

        self.buffer_nodes.append(node)

        return self._preprocess_state(self.state)



    def render_final(self,show_img = True,save_img = False,path_img = None):

        #create a figure

        fig = plt.figure(figsize=(10,7))

        fig.suptitle('HP MODEL 3DIMENSIONAL-PROTEIN FOLDING',c='blue',alpha=1)    

        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X',c='red')
        ax.set_ylabel('Y',c='red')

        colors = [0.8 if i == 'H' else 0 for i in self.seq[:self.state['id']+1]]

        k = list()

        for n in self.buffer_nodes:

            k.append(n.pos_array)

        k = np.array(k)

        z,y,x = k[:,0],k[:,1],k[:,2]

        ax.scatter(x,y,z,c=colors,s=150,alpha=1,cmap='plasma')

        for i in range(len(self.seq[:self.state['id']+1])-1):

            x1,y1,z1 = x[i],y[i],z[i]
            x2,y2,z2 = x[i+1],y[i+1],z[i+1]

            x_l = [x1,x2]
            y_l = [y1,y2]
            z_l = [z1,z2]

            if ((self.seq[i]=='H') & (self.seq[i+1]=='H')):

                ax.plot(x_l,y_l,z_l,c='green')

            else:
                ax.plot(x_l,y_l,z_l,c='green')   

        for (id_n1,id_n2) in self.buffer_relations:

            node1 = self.buffer_nodes[id_n1]
            node2 = self.buffer_nodes[id_n2]

            z_l = [node1.pos_array[0],node2.pos_array[0]]
            y_l = [node1.pos_array[1],node2.pos_array[1]]
            x_l = [node1.pos_array[2],node2.pos_array[2]]

            ax.plot(x_l,y_l,z_l,c='red',linestyle='--',alpha=0.8,linewidth=0.8)

        if save_img:

            dummy = plt.figure()
            
            new_mananger = dummy.canvas.new_manager(fig)

            assert path_img is not None, "Image needs a name"
        
            pickle.dump(fig,open(path_img,'wb'))

        if show_img:
            plt.show()
            print(f"Size of the sequence: {x_l.shape[0]}")


    def _preprocess_state(self,state:dict,npy=False):

        one_hot_actions = F.one_hot(torch.tensor(state['actions']),num_classes=7)

        last_state = torch.cat((self.seq_one_hot,one_hot_actions),1)

        if npy:
            
            return last_state.numpy()

        return last_state


    def _verify_solution(self):

        n,l,l = self.shape_grid

        collisions = 0

        no_valid_moves = 0

        for a in range(6):

            z,y,x = self.state['pos_array'] + MOVS[a,:]

            val = (z>=0) & (z<n) & (y>=0) & (y<l) & (x>=0) & (x<l)

            if not val:

                no_valid_moves += 1

            if self.grid[z,y,x]!=0:
                
                collisions += 1

        if no_valid_moves+collisions == 6:

            return False

        else:

            return True        


    def _update_buffer_relations(self,node: Node):

        if node.type=='H':

            for n in self.buffer_nodes:

                if node._calculate_adjacents(n)==1.0 and n.type=='H':
                    
                    n1,n2=node.id,n.id

                    self.buffer_relations.add((n1,n2))

    def _calculate_reward(self):

        return len(self.buffer_relations)          


       
