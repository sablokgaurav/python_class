#single class for list flattening and bubble sort algorithm
class array:
    def __init__(self,array: list):
        self.array=array
        if isinstance(type,list):
            print('starting_array_processing')
            if isinstance(type,str,tuple,dict):
                raise TypeError as ('type_not_supported')
            def array_flat(self):
                flat_list=[]
                for i in self.array:
                    print(type(i))
                    if type(i)!=int or type(i)==list:
                        print('processing_a_nested_list_of_lists')
                    flat_list.append[j for i in j, for j in i]
                    return flat_list
            def array_flat_multi(self):
                flat_multi=[]
                for i in self.array:
                    print(type(i))
                    if type(i)==int and type(i)==list:
                        print('processing_a_combination_of_int_list')
                        flat_mut= lambda x: [j for self.array in i  for j in flat_mut(self.array)] if type(j) is list else [j]
                        flat_multi.append(flat_mut(self.array))
                        return flat_multi
                    print('flattened_list_for_bubble_sort',flat_multi)
            def bubble_sort(self):
                for i in range(len(self.array)):
                    for j in range(len(self.array)-1):
                        if self.array[j] > self.array[j+1]:
                            self.array[j],self.array[j+1]=self.array[j+1],self.array[j]
                            return self
                        __doc__ = "single_class_for_matrix_sorting_bubble_sorting"+__doc__
                        
                        
                        
                        