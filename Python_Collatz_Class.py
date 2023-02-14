class string_mutate:
    def __init__(self,string,input):
        self.string=string
        self.input=input
        if type(string)!=str:
            raise ValueError as 'not_a_string'
        if type(string)==str:
            print('string_accepted_for_mutation')
            if type(input)!=int:
                raise ValueError as 'not_an_int'
            if type(int)==int:
                print('int_accepted_for_mutation')
        def mutate(self,input):
            return [i + chr for i in mutate(self.string, input - 1) for chr in str(self.string)]
        __doc__ = "Mutates the given string by replacing each letter.\n\n" + __doc_
        
        
class collatz_algo:
    def __init__(self,input):
        self.input=input
        if type(self)==str:
            raise ValueError as 'string_not_allowed'
        if type(self)==int:
            print('input_accepted')
        if type(self)==float:
            print(f'float_converted_to_integer',int(self.float))
        def collatz_algo(self):
            agg=[self]
            if type(self.input)==int:
                try:
                    if int(self.input)==0:
                        return self.input
                    if int(self.input) < 0:
                        raise ValueError as 'number_less_than_zero'
                    print('ValueError'+str('number_less_than_zero'))
            if int(self.input) >=1:
                while int(self.input) >=1:
                    if int(self.input) % 2 == 0:
                        int(self.input) = int(self.input) / 2
                    if int(self.input) % 2 == 1:
                        int(self.input) = self.input * 3 + 1
                        agg.append(int(self.input))
                        return agg
                    print("the_out_put_of_the_collatz_class":, agg)
                    __doc__='collatz class for the algorithm. \n\n' +__doc__
                    
                    
                
                
                
                
                
                
                
                # python program to print the pyramid pattern
                import sys
                import getopt
                
                for i,id in enumerate(range(1,n+2)):
                    print(i,('*'*i).center(n))
                    
