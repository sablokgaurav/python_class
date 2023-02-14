Class StringConversion:
    # a multi class using the static method for all string conversions
# convert the first letters
def to_title(s):
         return str(s).title()
    StringConversion.title=staticmethod(StringConversion.to_title)
# multi line ord conversion
def to_ord_multi(s):
         return [j for i in (list(map(ord,i)) for i in s.split()) for j in i]
    StringConversion.ord_multi=staticmethod(StringConversion.to_ord_multi)
# multi line upper conversion
def to_upper_multi(s):
       return [''.join(i) for i in (list(map(lambda n: chr(ord(n)-32),i))
                                    for i in (list(i.lower()) for i in list(s.split())))]
   StringConversion.upper_multi=staticmethod(StringConversion.to_upper_multi)
# multi line lower conversion
def to_lower_multi(s):
        return [''.join(i) for i in (list(map(lambda n: chr(ord(n)+32),i))
                                    for i in (list(i.upper()) for i in list(s.split())))]]
   StringConversion.lower_multi=staticmethod(StringConversion.to_lower_multi)