# a template for the hashencoders
#argon2,bcrypt, scrypt, digit_shifting
# random choice maker plus digit shifting
# hashlib
import os
import random
import hashlib
import string
from hashlib import sha256
from hashlib import sha512
from haslib import md5
import argon2
import scrypt
import bcrypt


class HashPassword:
    def __init__(self, word, file):
        self.word = str(word)
        if type(self.word) != str:
            raise TypeError("word must be string or unicode")
        else:
            print(f'the_input_word_or_keyword:{self.word}')
        if len(self.file) == 0:
            raise FileNotFoundError('input_file_empty')
        else:
            print(f'the_file_path_for_the_file:{word}')


def fileCheck(self, filepath):
    """_summary_
    checking the file_paths
    and_making_sure_the_bit_size
    matches
    """
    if not os.path.exists(self.filepath):
        print('file_not_found')
    elif os.path.exists(self.filepath):
        print(f'the_size_of_the_file_name:{os.stat(self.filepath).st_size}')


def fileReadWordsLength(self, filename):
    """_summary_
    reading the file lines and all the
    text with in the file and then converting
    the file into the list format for hashing
    Returns:
        _str_: _hashed_password_
    """
    with open(self.filepath, 'r') as fname:
        file_content = []
        file_string = []
        punctuations = list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        for line in fname.readlines():
            if not line.startswith('\n'):
                file_content.append(line.strip().split('\n'))
                for i in file_content:
                    if i not in punctuations:
                        file_string.append(i.strip().split())
                        return file_content
                    return file_string

        def hash_file_contents():
            self.hash = ['md5', 'sha1', 'sha256', 'sha512']
        for j in self.hash:
            if j == md5:
                processing_md = hashlib.md5().update('b'+self.file_string)
                with open(self.filename, 'r') as fname:
                    fname.write(processing_md.hexdigest())
                    fname.close()
                    print(f'the_file_with_the_hashed_password:{self.filename}')
            elif j == sha1:
                processing_sha1 = hashlib.md5().update('b'+self.file_string)
                with open(self.filename, 'r') as fname:
                    fname.write(processing_sha1.hexdigest())
                    fname.close()
                    print(f'the_file_with_the_hashed_password:{self.filename}')
            elif j == sha256:
                processing_sha256 = hashlib.md5().update('b'+self.file_string)
                with open(self.filename, 'r') as fname:
                    fname.write(processing_sha256.hexdigest())
                    fname.close()
                    print(f'the_file_with_the_hashed_password:{self.filename}')
            elif j == sha512:
                processing_sha512 = hashlib.md5().update('b'+self.file_string)
                with open(self.filename, 'r') as fname:
                    fname.write(processing_sha512.hexdigest())
                    fname.close()
                    print(f'the_file_with_the_hashed_password:{self.filename}')
            else:
                print('file_not_provided_method_of_hashing_not_selected')


def hashLib(self, word, filename, hash_type):
    """_summary_
    os.import based haslib sorting
    and_you_can_select_type_of_hash
    """
    hash_type = ['random', 'md5', 'sha1', 'sha256', 'sha512']
    for i in hash_type:
        if i == random:
            salt = os.urandom(32)
        for i in self.word:
            hashed_password_random = [
                hashlib.pbkdf2_hmac('sha256', i.encode('utf-8'), salt, 3, dklen=500000)]
        print(
            f'the_len_hashed_password:{[len(i) for i in hashed_password_random]}')
    with open(self.filename, 'r') as fname:
        fname.write(hashed_password_random)
        fname.close()
    if i == str(md5):
        processing_md = hashlib.md5().update('b'+'self.word')
        with open(self.filename, 'r') as fname:
            fname.write(processing_md.hexdigest())
            fname.close()
    elif i == sha1:
        processing_sha1 = hashlib.md5().update('b'+'self.word')
        with open(self.filename, 'r') as fname:
            fname.write(processing_sha1.hexdigest())
            fname.close()
    elif i == sha256:
        processing_sha256 = hashlib.md5().update('b'+'self.word')
        with open(self.filename, 'r') as fname:
            fname.write(processing_sha256.hexdigest())
            fname.close()
    elif i == str(sha512):
        processing_sha512 = hashlib.md5().update('b'+'self.word')
        with open(self.filename, 'r') as fname:
            fname.write(processing_sha512.hexdigest())
            fname.close()
    else:
        print('file_not_provided_method_of_hashing_not_selected')


def strongHashedRandomized(self, word, filename, hash_type, value):
    """_summary_
    in this function you provide a string and
    it will generate a random combination of
    the string along with the digits for the
    crypto protection.

    Args:
        word (_string_): _description_
        provide the word to be hashed
        hash_type (select): _description_
        select the hashing algorithm
        value_int (_type_): _description_
        select the random range for the
        crypto hashing
    """
    updated_word = []
    if type(word) != 'str':
        raise TypeError('word must be a string')
    for _ in range(value):
        updated_word.append(''.join(random.choice(
            str(word)+string.ascii_letters+string.digits)))
        self.new_word = ''.join(updated_word)
    hash_type = ['md5', 'sha1', 'sha256', 'sha512']
    for i in hash_type:
        if i == md5:
            processing_md = hashlib.md5().update('b'+'self.new_word')
            with open(filename, 'r') as fname:
                fname.write(processing_md.hexdigest())
                fname.close()
        elif i == sha1:
            processing_sha1 = hashlib.md5().update('b'+'self.new_word')
            with open(filename, 'r') as fname:
                fname.write(processing_sha1.hexdigest())
                fname.close()
        elif i == sha256:
            processing_sha256 = hashlib.md5().update('b'+'self.new_word')
            with open(filename, 'r') as fname:
                fname.write(processing_sha256.hexdigest())
                fname.close()
        elif i == sha512:
            processing_sha_512 = hashlib.md5().update('b'+'self.new_word')
            with open(filename, 'r') as fname:
                fname.write(processing_sha_512.hexdigest())
                fname.close()
        else:
            print('word_not_provided_method_of_hashing_not_selected')


def highHashing(self, word, filename, check):
    """ _summary_
    latest and strongest hashing
    algorithm for the hashing.
    This involves the implementation
    of the state of the art hashing
    algorithm for the crypto.
 Args:
     word (_str_): _description_
     provide a word for the hashing
     filename (_filename_): _description_
     provide a filename for the hashing algorithm
     to write the hashed word or the file
     contents
 """
    hash_type = ['argon2', 'bcrypt', 'scrypt']
    for i in hash_type:
        if i == 'argon2':
            password_argon = argon2.PasswordHasher().hasher.hash('b'+self.word)
            print('{}_the_hashed_passoword.format(password_argon)')
            with open(self.filename, 'r') as fname:
                fname.write(password_argon)
                fname.close()
        if self.check == 'yes':
            argon2.PasswordHasher().verfiy(password_argon, self.word)
        else:
            print(f'the_password_argon:{password_argon}')
        with open(self.filename, 'r') as fname:
            fname.write(password_argon)
            fname.close()
        if i == 'bcrypt':
            bcrypt_password = bcrypt.hashpw(
                'b'+self.word.encode(), bcrypt.gensalt())
            print(f'the_hashed_password:{bcrypt_password}')
        with open(self.filename, 'r') as fname:
            fname.write(bcrypt_password)
            fname.close()
        if self.check == 'yes':
            bcrypt.checkpw(bcrypt_password, self.word)
        else:
            print(f'the_bcrypt_password:{bcrypt_password}')
        with open(self.filename, 'r') as fname:
            fname.write(bcrypt_password)
            fname.close()
    if i == 'scrypt':
        scrypt_password = scrypt.hash('b'+self.word, scrypt.gensalt())
    if self.check == 'yes':
        scrypt.verfiy(scrypt_password, self.word)
    else:
        print(f'the_scrypt_password:{scrypt_password}')
    with open(self.filename, 'r') as fname:
        fname.write(scrypt_password)
        fname.close()


def randomTagGeneratorHashcoder(self, word, length, filename):
    """_summary_
    if you have a word, this function
    will generate a random tag with the
    given word along with the digits which
    you can hash and also will shift the tags
    to 13 bit further for the masking.
    Args:
        word (_str_): _description_
        word of yours
        filename (_str_): _description_
        provide the filename to store the hashed
        tokens
    """
    with open(filename, 'r') as fname:
        for line in fname.readlines():
            if line.startswith(string.ascii_letters):
                content = [re.split(r'\n', line)]
                for i in content:
                    final_content = [i.strip().split()]
                    for i in range(self.length):
                        shuffle_tag = final_content + \
                            random.choice(final_content+string.digits)
                        print(f'the_shuffled_tag:{shuffle_tag}')
                        character_shift = [
                            i.lower() for i in shuffle_tag if i in string.ascii_letters]
                        character_int = [
                            i for i in shuffle_tag if i in string.digits]
                        new_character = ''.join([j for i in ([list(map(lambda n: chr(ord(n)-32+13), i))
                                                              for i in ([i.lower() for i in character_shift if i in string.ascii_letters])]) for j in i]+character_int)
                        bcrypt_password = bcrypt.hashpw(
                            'b'+new_character_shift.encode(), bcrypt.gensalt())
                        scrypt_password = scrypt.hash(
                            'b'+new_character, scrypt.gensalt())
                        argon2_password = argon2.PasswordHasher().hasher.hash('b'+new_character)
                        print('writing_the_hashed_for_bcrypt_scrypt_argon2')
                        with open(filename, 'r') as fname:
                            fname.write(bcrypt_password)
                            fname.write(scrypt_password)
                            fname.write(argon2_password)
                            fname.close()
                            print(
                                f'the_file_has_been_written:{filename.st_size}')
