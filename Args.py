import sys, getopt

def arguments(argv=sys.argv):
    argIn = {"k": 1, "iterations": 100, "population": 50, "dataset": "Iris"}
    try:
        opts, args = getopt.getopt(argv[1:], "hk:p:i:d:",
                                   ["kset=", "iterations=", "population=", "dataset="])
        for opt, arg in opts:
            if opt == "-h":
                print(argv[:1] + " -k <set> -p <population> -i <iterations> -d <dataset>")
                sys.exit()
            elif opt in ("-i", "--iterations"):
                argIn['iterations'] = int(arg)
            elif opt in ("-k", "--kset"):
                argIn['k'] = int(arg)
            elif opt in ("-p", "--population"):
                argIn['population'] = int(arg)
            elif opt in ("-d", "--dataset"):
                argIn['dataset'] = arg

    except getopt.GetoptError:
        print(argv[:1] + " -k <set> -p <population> -i <iterations> -d <dataset>")
        sys.exit(2)
    except:
        sys.exit(2)
    return argIn