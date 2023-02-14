class binary:
    '''
    reverse a binary int provided as a string based on the base class
    of the input
    '''
    def __init__(self,num):
        self.num=num
        print('binary_number_is_'+str(self.num))
        print('this_class_takes_binary_number_as_input')
        def binary_class(self):
            if type(num)==list:
                raise TypeError('only_takes_binary_number'+str(self.num))
            if type(num)==int:
                raise ValueError('only_takes_binary_number_as_string'+str(self.num))
            if type(num)==str:
                 inverse=[]
                var1,var2=0,abs(self.num)
            while var2:
                var1 = var1*10 + var2%10
                var2 //= 10
            inverse.append(-var1 if self.num < 0 else var1)
            return inverse
        print(f'the_reverse_of_the_given_int_as_a_base_class:'{}.format(inverse))        
        def string_class(self):
            if type(num)==list:
                raise TypeError('binary_number_is_'+str(self.num))
            if type(num)==int:
                raise ValueError('binary_number_is_'+str(self.num))
            return int(''.join(map(str,(list(map(int,''.join(list(str(a)))))[::-1]))))
        print(f'the_reverse_of_the_given_int_as_a_string:'{}.format(int(''.join(map(str,(list(map(int,''.join(list(str(a)))))[::-1])))))
              __doc__='''A_single_class_which_will_inverse_number_based_on_both_the_base_class_as_well_as_the_string_class'''
