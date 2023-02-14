class list_utils:
    def __init__(self,input:list) -> list:
        self.input=input
        print(f'{self.input}')
        def list_even(self): # taking out positive indexes from list for the iterations
            for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                else:
                        return [self.input[i::2] for i in range(len(self.input))]
                print(f'the_length_of_{}_even_array:'.format(len([self.input[i::2] for i in range(len(self.input))])))
        def list_odd(self): # taking out odd indexes from list for the iterations
            for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                else:
                        return [self.input[i::3] for i in range(len(self.input))]
                 print(f'the_length_of_{}_odd_array:'.format(len([self.input[i::3] for i in range(len(self.input))])))
class list_window(list_utils): #taking out the windowed iterations from the list
    super().__init__()
    def list_select(self, num):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                    if type(i)==list and len(self.input) >=1:
                        return [[self.input[i:i+num]] for i in range(len(self.input)-(num-1))]
                print(f'your_list_is_windowed_accordingly',num)          
class transpose_list(list_utils): #transposing the nested list
    super().__init__()
    def transpose(self):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                    if type(i)==list and len(self.input) >=1:
                        return [list[i]for i in zip(*self.input)]
                    print(f'your_list_has_been_transposed'{}.format(len([list[i]for i in zip(*self.input)])))
class insertion_sort(list_utils): # insertion sort the list
    super().__init_()
    def insertion_sort(self):
        if len(self.input)==1:
            return 1
        else:
            print('list_can_be_insert_sort')
        for i in range(1,len(self.input)):
             if self.input[i-1] > self.input[i]:
                 self.input[i-1],self.input[i]=self.input[i],self.input[i-1]
                 return self.input
class dynamic_sum(list_utils): #sum of the list
    super().__init__()
    def sum_list(self):
        if len(self.input)==1:
            return self.input[0]
        if len(self.input)==0:
            return 0
        else:
           return self.input[0] + sum_list(self.input[1:])
    print(f'the_sum_of_the_list_element':{}.format(self.input[0] + dynamic_sum(self.input[1:])))
class reverse(list_utils): #reversing every list of the nested list
    super().__init__()
    def reverse(self):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                    if type(i)==list and len(self.input) >=1:
                        return [self.input[::-1]for i in range(len(self.input))]
                    print(f'The_unique_count_of_elements_in_array:'{}.format([self.input.count(i) for i in unique]))
class frequency_cal(list_utils): #frequency calculation of the nested list
    super().__init__()
    def frequency_cal(self):
        count = []
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                    if type(i)==list and len(self.input) >=1:
                        unique=set(self.input)
                        return [self.input.count(i) for i in unique]
                        print(f'The_unique_count_of_elements_in_array:'{}.format([self.input.count(i) for i in unique]))                  
class median_even(list_utils): #median of the list if it is even
    super().__init__()
    def median(self):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                    if len(self.input)%2==1:
                        print(f'the_array_is_odd:'{}.format(len(self.input)%2))
                        median_value=[]
                        if len(self.input)%2==0:
                            n=len(self.input)//2
                            median_value.append([sum(i)/2 for i in ((i[0][-1],i[1][0]) for i in ((i[:n],i[n:]) for i in [sorted(self.input)]))])
                            return median_value
                        print(f'The_median_of_given_array:'{}.format(median_value))
class median_odd(list_utils): #median of the list if it is odd
    super().__init__()
    def median(self):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError as type_not_accepted
                    print(f'TypeError'+str(type_not_accepted))
                    if len(self.input)%2==0:
                        print(f'the_array_is_even:'{}.format(len(self.input)%2))
                        median_value=[]
                        if len(self.input)%2==0:
                            median_value.append([(i) for i,id in enumerate(sorted(self.input),1) if (int((len(sorted(self.input))+1)//2))==i])
                            return median_value
                        print(f'The_median_of_given_array:'{}.format(median_value))
                    __doc__'a_simple_class_which_do_all_calculations_on_1D_arrays_as_well_as_on_2D_arrays'+__doc__""
                    
                    
                  