age = raw_input('What is your age? ')
if age >= '18':
    print 'You have authentication for facebook'
elif age <= '12':
    print 'Waiting for the 18'
else:
    print 'Please provide the age in numeric value'
     
