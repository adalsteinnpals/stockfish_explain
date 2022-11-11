from datetime import datetime

def calculatedaysbetweendates(date1, date2):
    # Calculate the number of days between two dates
    # date1 and date2 are strings in the format YYYY-MM-DD
    # Return an integer
    # YOUR CODE HERE
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')
    return abs((date2-date1).days)

# Test your function with the following dates   
date1 = '2013-10-01'
date2 = '2013-11-01'
print(calculatedaysbetweendates(date1, date2))
