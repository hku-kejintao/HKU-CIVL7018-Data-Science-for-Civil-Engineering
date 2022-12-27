##########################################
##        Written by Wang Chen          ##
##    E-mail: wchen22@connect.hku.hk    ##
##########################################


import numpy as np
from time import sleep
import random
from tqdm import tqdm
import argparse

# parameters
def parse_args():
    parser = argparse.ArgumentParser(description='Q-table')
    parser.add_argument('--method',
                        help = 'which method will be used to update the Q-table',
                        type = str,
                        default = 'Bellman')
    parser.add_argument('--EpochNum',
                        help = 'the number of training epoches',
                        type = int,
                        default = 200)
    
    args = parser.parse_args()

    return args


'''The object of Q-table'''
class Q_table():
    def __init__(self,
                 table_height = 4,
                 table_width = 4,
                 target_location = [(2,2)],
                 obstacle_location = None,
                 stride_reward = -1,
                 boundary_value = -99,
                 target_reward = 100,
                 obstacle_reward = None,
                 lr = 0.1,
                 discount = 0.9
                 ):
        # Initialize the table, and 4 channels represent Up, Right, Down, and Left, respectively
        self.actions = ['up', 'right', 'down', 'left']
        self.obstacle_location = obstacle_location
        
        self.table_height = table_height
        self.table_width = table_width
        self.table = np.zeros([4, table_height, table_width])
        
        # Initialize rewards
        self.boundary_reward = boundary_value
        self.obstacle_reward = obstacle_reward
        self.target_reward = target_reward
        self.stride_reward = stride_reward

        # Constrain the boundary
        self.table[0, 0, :] = self.boundary_reward + self.stride_reward
        self.table[1, :, -1] = self.boundary_reward + self.stride_reward
        self.table[2, -1, :] = self.boundary_reward + self.stride_reward
        self.table[3, :, 0] = self.boundary_reward + self.stride_reward
        
        # Initialize the reward, and each stride reward is -1
        self.Reward = np.ones((table_height, table_width)) * stride_reward
        assert len(target_location) == 1 # We assume there only exists 1 target
        self.target_location = target_location[0]
        
        # Set target reward
        self.Reward[self.target_location[0], self.target_location[1]] += target_reward
        
        # Set obstacle reward
        if obstacle_location is not None:
            assert obstacle_reward is not None
            for obs in obstacle_location:
                self.Reward[obs[0], obs[1]] += obstacle_reward
        
        print('Reward:')
        print(self.Reward)
        print('='*40)
        
        self.cur_y, self.cur_x = 0, 0
        self.lr = lr
        self.discount = discount
    
    
    # Update the Q-table through Bellman function (Stochastic Gradient Descent)
    # params: the current position and action
    def update_Q_table_Bellman(self, pos_y, pos_x, action):
        self.cur_y, self.cur_x = pos_y, pos_x
        
        # Update the next location according to the action
        next_y, next_x = self.get_next_position(pos_y, pos_x, action)
        
        # If the next location is in the table, i.e., the action is valid, we update the Q-value
        # Otherwise, the Q-value of the action at the current position (state) is the boundary_value
        if next_y is not None:
            max_next_pos_Q_val = np.max(self.table[:, next_y, next_x])
            delta_Q = self.Reward[next_y, next_x] + self.discount * max_next_pos_Q_val \
                      - self.table[action, self.cur_y, self.cur_x]
            # Update Q-table
            self.table[action, self.cur_y, self.cur_x] += self.lr * delta_Q
 
    
    # Update the Q-table through Monte Carlo sample
    # params: the current position and action
    def update_Q_table_MonteCarlo(self, pos_y, pos_x, action, sample_times = 1000):
        # We don't update the target location
        if (pos_y, pos_x) == self.target_location:
            return
        # If the next location is in the table, i.e., the action is valid, we update the Q-value
        # Otherwise, the Q-value of the action at the current position (state) is the boundary_value
        next_y, next_x = self.get_next_position(pos_y, pos_x, action)
        if next_y is None:
            return
        # The agent touches the obstacle
        if (next_y, next_x) in self.obstacle_location:
            self.table[action, pos_y, pos_x] = self.obstacle_reward + self.stride_reward
            return
        # The agent touches the diamond
        if (next_y, next_x) == self.target_location:
            self.table[action, pos_y, pos_x] = self.target_reward + self.stride_reward
            return
        
        # If the system doesn't terminate
        Reward = 0 # Total rewards
        
        for idx in range(sample_times):
            next_y_tmp, next_x_tmp = next_y, next_x
            #while True:
            for i in range(10000): # For safety, we assume the agent move 1000 strides at most
                action_tmp = random.randint(0, 4)
                next_y_tmp, next_x_tmp = self.get_next_position(next_y_tmp, next_x_tmp, action_tmp)
                Reward += self.stride_reward # reward for each action
                if next_y_tmp is None:
                    Reward += self.boundary_reward
                    break # the agent does not find the target
                if (next_y_tmp, next_x_tmp) in self.obstacle_location:
                    Reward += self.obstacle_reward
                    break # the agent does not find the target
                if (next_y_tmp, next_x_tmp) == self.target_location:
                    Reward += self.target_reward # the agent finds the target
                    break
        
        # Update Q-table
        self.table[action, pos_y, pos_x] = Reward / sample_times
            
    
    
    # Get the next location according to the action
    def get_next_position(self, cur_y, cur_x, action):
        if action == 0: # Up
            if cur_y > 0: # Make sure the next location is in the table
                next_y, next_x = cur_y - 1, cur_x
                return next_y, next_x
        elif action == 1: # Right
            if cur_x < self.table_width -1:
                next_y, next_x = cur_y, cur_x + 1
                return next_y, next_x
        elif action == 2: # Down
            if cur_y < self.table_height - 1:
                next_y, next_x = cur_y + 1, cur_x
                return next_y, next_x
        elif action == 3: # Left
            if cur_x > 0:
                next_y, next_x = cur_y, cur_x - 1
                return next_y, next_x
        
        return None, None # The next position is not in the table
    
    
    # Show the actions of the agent from the given position to the target position
    def show_actions(self, pos_y = 0, pos_x = 0):
        pos_y, pos_x = pos_y, pos_x
        # For simplicity, we use the array to show the actions
        whole_map = np.zeros([self.table_height, self.table_width])
        whole_map[pos_y, pos_x] = 1
        print(whole_map)
        
        total_steps = (self.table_width + self.table_height) * 3 # We constrain the total steps
        for i in range(total_steps):
            action = np.argmax(self.table[:, pos_y, pos_x])
            print('action: ', self.actions[action])
            
            pos_y, pos_x = self.get_next_position(pos_y, pos_x, action)
                    
            # Check if the next position is in the table
            if pos_x is None or (pos_y, pos_x) in self.obstacle_location:
                print('It seems that the agent is not intelligent enough to find the target')
                return
            
            sleep(2) # We sleep 2s for watching the results
            whole_map[pos_y, pos_x] = 1
            print('='*40)
            print(whole_map)
            if (pos_y, pos_x) == self.target_location: # Find the target position
                print('='*40)
                print('Good job!')    
                return
        print('It seems that the agent is not intelligent enough to find the target')


def main():
    args = parse_args()

    # Initialize the Q-table
    table_height = 4
    table_width = 4
    Qtable = Q_table(table_height = table_height,
                     table_width = table_width,
                     target_location = [(2,2)],
                     obstacle_location = [(2,1),(1,2)],
                     stride_reward = -1,
                     boundary_value = -100,
                     target_reward = 100,
                     obstacle_reward = -100,
                     lr = 0.1,
                     discount = 0.9
                     )
    
    if args.method == 'Bellman':
        # Train the agent and update the Q-table
        total_epochs = args.EpochNum
        for epoch in tqdm(range(total_epochs), desc = 'Train the agent'):
            for pos_x in range(table_width):
                for pos_y in range(table_height):
                    for action in range(4):
                        Qtable.update_Q_table_Bellman(pos_y, pos_x, action)
        
        # Show the results
        print('Updated Q-table:')
        print(Qtable.table)
        print('='*40)
        Qtable.show_actions()
                    
    elif args.method == 'MC':
        Qtable.target_reward = 10000
        for pos_x in tqdm(range(table_width), desc = 'Train the agent by Monte Carlo sampling'):
            for pos_y in range(table_height):
                for action in range(4):
                    #Qtable.update_Q_table_Bellman(pos_y, pos_x, action)
                    Qtable.update_Q_table_MonteCarlo(pos_y, pos_x, action, sample_times=100000)
            
        # Show the results
        print('Updated Q-table:')
        print(Qtable.table)
        print('='*40)
        Qtable.show_actions()

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()