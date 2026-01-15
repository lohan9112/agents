def decorator(func):
    def wrapper():
        func()
        return 10
    return wrapper


@decorator
def say_hello():
    print("Hello, world!")

def say_greeting():
    print("How are you?")

decora_function = decorator(say_greeting)
decora_function2 = decorator(say_hello)