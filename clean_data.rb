require 'csv'
data = CSV.read('voice_to_do_list.csv', headers: true)

cleaned_data = data.reject { |row| row.to_h.values.any?(&:nil?) }

data.each do |row|
  row.each do |field, value|
    row[field] = 'default_value' if value.nil?
  end
end

unique_data = data.uniq { |row| row['column_name'] }

data.each do |row|
    row['numeric_column'] = row['numeric_column'].to_i
  end

  CSV.open('voice_to_do_list_cleaned.csv', 'wb') do |csv|
    csv << data.headers
    data.each do |row|
      csv << row
    end
  end