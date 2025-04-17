def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def check_even_odd(num):
    return "Even" if num % 2 == 0 else "Odd"

if __name__ == "__main__":
    try:
        number = int(input("Enter a number: "))
        print(f"\n🔢 Factorial of {number} is: {factorial(number)}")
        print(f"🧮 {number} is an {check_even_odd(number)} number.")
    except ValueError:
        print("❌ Please enter a valid integer.")

