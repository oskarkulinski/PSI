import string

class article:
    def __init__(self, name : str, authors : list, word_count : int, contents : str):
        self.name = name
        self.authors = authors
        self.word_count = int(word_count)
        self.contents = contents

    def get_authors_short(self):
        names = list()
        for author in self.authors:
            first_name, space, last_name = author.partition(" ")
            names.append(first_name[0] + ". " + last_name)
        return names

    def get_name(self):
        return self.name

    def get_average_word_lenght(self):
        totalLength = 0
        for l in self.contents:
            if l.isalpha():
                totalLength += 1
        return totalLength//self.word_count

    def get_raw_words(self):
        return self.contents.casefold().translate(str.maketrans("", "", string.punctuation)).split(" ")