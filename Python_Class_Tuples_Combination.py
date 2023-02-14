class tuple_list:
    def __init__(self,input : list([list])) -> list:
        self.input=input
        print(f'the_list_tuple_merging_class')   
    def make_tuple(self):
        for i in self.input:
                if type(i)==list:
                        raise TypeError('type_supported')
                        print('type_accepted')
                if type(i)==tuple:
                        raise TypeError ('type_not_supported')
                if type(i)==int:
                            raise TypeError('type_not_supported')
                if type(i)==list:
                        output=[]
                        output.append([([(i,j) for i,j in zip(*self.input)])])
                        return output   
    def combinations_tuple(self,arr1: list[list]) -> list:
        for i in self.input:
            for j in arr1:
                if type(i)==tuple:
                    raise TypeError('type_not_supported')
                    print('type_not_accepted')
                if type(j)==tuple:
                        raise TypeError('type_not_supported')
                        print('type_not_accepted')
                if type(i)==set:
                    raise TypeError('type_not_supported')
                    print('type_not_accepted')
                    if type(j)==set:
                        raise TypeError('type_not_supported')
                        print('type_not_accepted')
                if type(i)==list:
                    if type(j)==list:
                        output=[]
                        output.append([([(i,j) for i in self.input for j in self.input.arr1])])
                        return output
                '''A_tuple_class_combinations_iterations_tuples_dict_iterations'''