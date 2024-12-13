def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

class Calculator:
    def __init__(self):
        self.history = []

    def multiply_numbers(self, a, b):
        result = a * b
        self.history.append(f"Multiplication: {a} * {b} = {result}")
        return result

    def divide_numbers(self, a, b):
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        result = a / b
        self.history.append(f"Division: {a} / {b} = {result}")
        return result

    def show_history(self):
        for entry in self.history:
            print(entry)
