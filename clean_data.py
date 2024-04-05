import csv
import os

def clean_csv(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=';')
            headers = next(reader)  # This might raise StopIteration if the file is empty
            temp_filename = output_filename + '.tmp'

            with open(temp_filename, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile, delimiter=';')
                command_index = headers.index('command')
                writer.writerow(headers)

                for row in reader:
                    row[command_index] = row[command_index].replace('Olly', 'Ora')
                    row[command_index] = row[command_index].replace('olly', 'ora')
                    writer.writerow(row)

        # Replace the original file with the modified one
        os.replace(temp_filename, output_filename)
        print('CSV file cleaned successfully.')

    except StopIteration:
        print("The input file is empty or the header is missing.")

if __name__ == '__main__':
    input_filename = 'cleaned_data.csv'  # Ensure this is the correct path
    output_filename = 'final_cleaned_data.csv'  # It's safer to output to a different file
    clean_csv(input_filename, output_filename)
