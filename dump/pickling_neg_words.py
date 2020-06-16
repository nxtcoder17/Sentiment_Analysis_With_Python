import pickle

f = open ("neg_words.txt", "rt")

neg_words = set()
for line in f:
    if line.startswith (";") or line.isspace():
        continue
    else:
        neg_words.add (line[:-1])
        print (line[:-1])

print (", ".join (neg_words))

pickle.dump (neg_words, open("neg_words.pkl", "wb"))

