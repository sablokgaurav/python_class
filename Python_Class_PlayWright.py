# playwright class from webdriver to tag generation
class PlayWright:
    def __init__(self, url, name:str):
        self.url = url
        self.name = name
        print(f'the_playwright class:{self.url},{self.name}')

class Launch(Playwright):
    super().__init__()

    def launch(self,webkit,browser,page, regex):
        with sync_playwright as playwright:
            self.url = url
            self.webkit = playwright.webkit()
            self.browser = browser.new_page()
            self.page = browser.new_page()
            self.page = self.page.title()
            self.page.goto(self.url)
            self.content = self.page.content('regex')
            self.tag = [{i, id} for i, id in enumerate(self.content)]
            print(f'the_tags_present_in_the_webpage:{self.tag}')
            print(self.browser.close())

class LocateTags(Playwright):
    super().__init__()

    def locate(self, pattern):
        with sync_playwright as playwright:
            self.locate = self.page.locator('pattern').all()
            print(f'the_count_of_the_tags:{[len(self.locate)]}')
            print(self.browser.close())

class CleanPattern(Planywright):
    super().__init__()

    def clean(self):
        punctuations = ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        dictionary_marks=[chr(c+65-32) for c in range(26)]
        clean_tags = []
        for i in self.locate:
            if i not in punctuations and i not in dictionary_marks:
                clean_tags.append(i)
                return clean_tags
            print(f'the_clean_text_without_punctuations:{len(self.clean_tags)}')
    def count_words(self):
        punctuations = ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        count_marks = [[chr(c+65).lower() for c in range(26)] + [[chr(c+65).upper() for c in range(26)]]]
        count_tags=[]
        letter_count=[]
        for i in self.clean.tags:
            if i not in punctuations:
                count_tags.append([map(lambda n:list(n),i) for i in self.clean.tags.split()])
                letter_count.append(
                        {
                         (i[0], count_tags.count(i))
                           for i in count_tags 
                           for j in count_marks
                           if i[0] in count_marks
                           } )
                #letter_count.append(set([(i[0],count_tags.count(i)) for i in count_tags 
                #                                for j in count_marks if i[0] in count_marks]))
                print(f'the_length_of_the_total_tags:{len(count_tags)}')
                print(f'the_tags_counts_starting_with_specific_alphabets:{letter_count}')
                return self.count_tags
            return self.letter_count

class CleanContent(Planywright):
    super().__init__()

    def clean(self):
        punctuations = ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        dictionary_marks=[chr(c+65-32) for c in range(26)]
        clean_tags = []
        for i in self.content:
            if i not in punctuations and i not in dictionary_marks:
                clean_tags.append(i)
                return self.clean_tags
            print(f'the_clean_text_without_punctuations:{len(self.clean_tags)}')
    def count_words(self):
        punctuations = ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']
        count_marks = [[chr(c+65).lower() for c in range(26)] + [[chr(c+65).upper() for c in range(26)]]]
        count_tags=[]
        letter_count=[]
        for i in self.clean.tags:
            if i not in punctuations:
                count_tags.append([map(lambda n:list(n),i) for i in self.clean.tags.split()])
                letter_count.append(
                        {
                         (i[0], count_tags.count(i))
                           for i in count_tags 
                           for j in count_marks
                           if i[0] in count_marks
                           } )
                print(f'the_length_of_the_total_tags:{len(count_tags)}')
                print(f'the_tags_counts_starting_with_specific_alphabets:{letter_count}')
                return self.count_tags
            return self.letter_count
                
class Classify(Playwright):
    super().__init__()

    def classify(self,number:int):
        length = list(self.clean.tags)
        for i in self.clean_tags:
            if i.startswith(string.ascii_uppercase) or i.startswith(string.ascii_lowercase):
                print(f'the_clean_tags_accepted_for_classification_starting_with
                      _lower_alphabets: {[self.clean.tags.count(i) for i in self.clean.tags if i.startswith(string.upper_case)]}')
                print(f'the_clean_tags_accepted_for_classification_starting_with
                      _upper_alphabets: {[self.clean.tags.count(i) for i in self.clean.tags if i.startswith(string.lower_case)]}')
            tag_join = ''.join(self.clean.tags)
            tag_classify = [(tag_join[i:i+int(number)])
                            for i in range(len(tag_join)-(i) + 1)]
            print(f'the_length_of_the_total_tags:{len(tag_classify)}')
            return len(tag_join), len(tag_classify)
