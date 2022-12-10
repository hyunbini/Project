from gensim.models import KeyedVectors
import random

#Hyperparameter
p = 11
q = 17
n = p * q
tot = (p - 1) * (q - 1)
def gcd(n1, n2):
    while n2 != 0:
        n1, n2 = n2, n1%n2
    return n1
def publickey(tot):
    e = 2
    while e<tot and gcd(e, tot) != 1:
        e += 1
    return e
def privatekey(public):
    d = 1
    while (public * d) % tot != 1 or d == public:
        d += 1
    return d
def encrypt():
    message = input("Enter the message  : ")
    message_list = list(message)
    ascii_arr = []
    result_arr = []
    result_message = ""

    public = publickey(tot)
    secret = privatekey(public)

    for character in range(0, len(message_list)):
        ascii_arr.append(ord(message_list[character]))

    for i in range(0, len(ascii_arr)):
        res = ((ascii_arr[i]**public)%n)
        result_arr.append(res)

    for j in range(0, len(result_arr)):
        result_message += (chr(result_arr[j]))

    print("cipher message : {}".format(result_message))
    print("N : {}".format(n))
    print("Public key : {}".format(public))
    print("Secret key : {}".format(secret))
    return result_message


def decrypt(text):
    cipher_text = (text)
    text_list = list(cipher_text)
    result_ascii_arr = []
    de_result_arr = []
    text_arr = []
    result_message = ""
    public = publickey(tot)
    secret = privatekey(public)
    for character in range(0, len(text_list)):
        result_ascii_arr.append(ord(text_list[character]))
    for i in range(0, len(result_ascii_arr)):
        res = (result_ascii_arr[i]**secret)%n
        de_result_arr.append(res)
    for j in range(0, len(de_result_arr)):
        text_arr.append(chr(de_result_arr[j]))
    for character in range(0, len(text_arr)):
        result_message += text_arr[character]          
    return result_message

def modeling(res, ans, trynum):
    model = KeyedVectors.load_word2vec_format('C:/Users/yckhb/Desktop/GoogleNews-vectors-negative300.bin',binary=True)
    num = model.similarity(res,ans)
    count = trynum
    if(count>=2):
        print("hint : " + str(model.most_similar(positive=res, topn = 3)))
    return num

def game_main():
    systemnumber = input("Welcome to the word analogy game. Enter 1 to create a problem and 2 to solve the problem and 3 to exit: ")
    problem = []
    try_problem = 0
    while(1):
        if(systemnumber == '1'):
            word = encrypt()
            problem.append(word)
            print(problem)
            systemnumber = input("Problem created successfully. Enter 1 to create more, 2 to solve, and 3 to exit: ")
        elif(systemnumber=='2'):
            problemmessage = random.choice(problem)
            answer = input("The cipher message is " + str(problemmessage) + " , Look at the cipher message and guess what the original word is and write it down: " )
            result = decrypt(problemmessage)
            number = 0
            while(number<=0.5):
                number = modeling(result,answer,try_problem)
                if(number>=0.5):
                    systemnumber = input("Correct. Answer is " + str(result) + " ,Enter 2 to solve more problems, 1 to create problems, and 3 to exit: ")
                else:
                    try_problem = try_problem+1
                    answer = input("It's wrong, try again to write word: ")
        elif(systemnumber == '3'):
            break
        else:
            systemnumber = input("It's error, try again: ")

game_main()