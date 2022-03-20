
def output_file(file_name, predicted_y_test):
    open(file_name, "w").close()
    file = open(file_name, "a")
    file.write("ImageId,Label\n")
    for i in range(len(predicted_y_test)):
        file.write(str(i+1) + "," + str(predicted_y_test[i]) + "\n")
    file.close()
