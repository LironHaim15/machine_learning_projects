import random

for i in range(1, 5):
    amount = pow(2, i)
    output_file = open("centssss" + str(amount) + ".txt", "w")
    for j in range(amount):
        output_file.write(
            str(round(random.uniform(0, 1), 4)) + " " +
            str(round(random.uniform(0, 1), 4)) + " " +
            str(round(random.uniform(0, 1), 4)) + "\n")
    output_file.close()
