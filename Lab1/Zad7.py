text = "Python"

new_text = " "
for l in text:
    if l.isalpha():
        new_text += l + " "

print(new_text)