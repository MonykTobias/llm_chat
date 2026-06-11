from agents import review_session

if __name__ == "__main__":
    path = input("Enter the path to the file: ")
    language = input("Enter the language: ")

    answer = review_session(path, language)
    print(answer)
