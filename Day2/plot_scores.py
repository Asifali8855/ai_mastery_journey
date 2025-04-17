import matplotlib.pyplot as plt

students = ["Ali", "Sara", "Asif", "Aisha", "John"]
scores = [85, 90, 78, 92, 88]

plt.figure(figsize=(8, 5))
plt.bar(students, scores, color='skyblue')
plt.xlabel("Students")
plt.ylabel("Scores")
plt.title("Student Test Scores")
plt.grid(True)
plt.tight_layout()

# Save and Show
plt.savefig("Day2/images/plot_output.png")  # Save as image
plt.show()  # Open window (if supported)

