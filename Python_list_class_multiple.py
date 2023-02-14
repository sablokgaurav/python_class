#class based solution for median of the nested arrays
class median_nested_even: 
    def __init__(self,arr:list) -> list:
        self.arr=arr
        if len(self.arr)==0:
            return []
        print(f'the_array_is_null:'{}.format(len(self.arr)))
    '''
    median of the nested list of lists if the nested list are even
    '''
    def median(self):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError
                    print(f'TypeError'+str('type_not_accepted'))
                    if len(self.input)%2==1:
                        print(f'the_array_is_odd:'{}.format(len(self.input)%2))
                        median_value=[]
                        if len(self.input)%2==0:
                            n=[len(i)//2 for i in self.input]
                            median_value.append([sum(i)/2 for i in ([(i[0][-1],i[1][0]) 
                                                    for i in ([([sorted(self.input[i]) 
                                                for i in range(len(self.input))][i][:n[i]],[sorted(self.input[i])
                                                                      for i in range(len(self.input))][i][n[i]:]) 
                                                                for i in (sorted(range(len([sorted(self.input[i]) 
                                                                       for i in range(len(self.input))]))))])])])
                            return median_value 
                        print(f'The_median_of_given_array:'{}.format(median_value))              
class median_nested_odd:
    def __init__(self,arr:list) -> list:
        self.arr=arr
        if len(self.arr)==0:
            return []
        print(f'the_array_is_null:'{}.format(len(self.arr)))
    '''
    median of the nested list of the lists are odd
    '''
    def median(self):
        for i in self.input:
                if type(i)==list:
                    print('type_accepted')
                    if type(i) != list:
                        raise TypeError 
                    print(f'TypeError'+str('type_not_accepted'))
                    if len(self.input)%2==1:
                        print(f'the_array_is_odd:'{}.format(len(self.input)%2))
                        median_value=[]
                        if len(self.input)%2==0:
                            median_value=[]
                            median_value.append(list(set([[sorted(self.input[i]) for i in range(len((self.input)))][i][j-1] 
                                             for i in range(len([sorted(self.input[i]) for i in range(len((self.input)))])) 
                                                                    for j in [(len(i)+1)//2 for i in [sorted(self.input[i]) 
                                                                                    for i in range(len((self.input)))]]])))
                            return median_value
                        print(f'the_median_value_of_the_array:'.format(median_value))
                        __doc__'''A_multilevel_class_for_arthimetic_median_estimation_nested_list '''
        