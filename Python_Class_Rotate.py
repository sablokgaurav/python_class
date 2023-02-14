from collections import deque                    
class rotate: 
    def __init__(self,input,num):
        '''
        A deque and a function class for rotation, inversion, and reversing.
        Minimizing the time complexity to O(n) as invoking deque
        instead of a list. 
        '''
        self.input = input
        self.num=num
        print(f'the_length_of_the_input_list:{}'.len(self.input))
    def rotate_single_node(self): # rotating a single list based on the number
        if type(self.input)!=list:
            raise TypeError('only_list_accepted_for_rotation')
        print(f'The_type_of_the_input:{type(self.input)}')
        var_deque=deque(self.input)
        rotate_deque=deque(self.input).rotate(self.num)
        return rotate_deque
    def rotate_multiple_node(self,num): # inplace rotation of the multiple windows
        if type(self.input)!=list:
            raise TypeError('only_list_accepted_for_rotation')
        print(f'The_type_of_the_input:{type(self.input)}')
        windowed=[]
        for i in range(len(self.input)):
            windowed.append([self.input[i:i+num] for i in range(len(self.input)-(num-1))])
            return windowed
        rotate_windowed=[]
        for j in range(len(windowed)):
            rotate_windowed.append(windowed[~j])
            return rotate_windowed
        print(f'the_rotate_multiple_node_of_the_input:{}'.format(rotate_windowed))
    def inverse_rotate(self,num): # make a windowed iterable and then inverse each nested window
        if type(self.input)!=list:
            raise TypeError('only_list_accepted_for_rotation')
        print(f'The_type_of_the_input:{type(self.input)}')
        inverse_windowed=[]
        for i in range(len(self.input)):
            inverse_windowed.append([self.input[i:i+num] for i in range(len(self.input)-(num-1))])
            return inverse_windowed
        inverse_rotate_windowed=[]
        for j in range(len(inverse_windowed)):
            inverse_rotate_windowed.append(inverse_rotate_windowed[~j][::-1])
            return inverse_rotate_windowed
        print(f'the_inverse_selection_of_the_input:{}'.format(inverse_rotate_windowed))
        __doc__=''' A_deque_class_and_function_based_class_for_the_list_rotation'''__doc__
        