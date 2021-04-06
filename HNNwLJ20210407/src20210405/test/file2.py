import file1 as f1

print('__name__ from file1 : {}'.format(f1.__name__))
print('__name__ from file2 : {}'.format(__name__))

if __name__ == '__main__':
    print('file2 is being run directly')
else:
    print('file2 is being imported')