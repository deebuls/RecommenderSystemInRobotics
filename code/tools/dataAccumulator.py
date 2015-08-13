#!/home/deebuls/anaconda/bin/python

import sys, getopt
import os
import glob
import json

def main(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfolder> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            print 'inputfolder : The folder in which all the json files are stored'
            print '              default : /tmp'
            print 'outputfile : The name of the experiments. A folder will be created '
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    if (outputfile == ''):
        print 'ERROR : Please specify experiment name'
        sys.exit()
    if (inputfile == ''):
        inputfile = '/tmp'
    print 'Input file is "', inputfile
    print 'Output file is "', outputfile

    return inputfile, outputfile

if __name__ == "__main__":
    inputfile, outputfile = main(sys.argv[1:])

    print 'Creating Directory ', outputfile
    if (0 != os.system("mkdir " + outputfile)):
        exit();
    print 'Copying all json files from ', inputfile, " to ", outputfile
    if (0 != os.system("mv " + inputfile + "/" + "*.json" + " " + outputfile)):
        exit()
        
    read_files = sorted(glob.glob("./" + outputfile + "/" + "i*.json"))
    output_list = []

    print 'reading all the initial files'
    for f in read_files:
        with open(f, "rb") as infile:
            output_list.append(json.load(infile))

    print 'creating the combined json INITIAL file'
    with open("./" + outputfile + "/" + "INITIAL.json", "wb") as outfile:
        json.dump(output_list, outfile)

    del read_files
    print 'reading all the FINAL files'
    read_files = sorted(glob.glob("./" + outputfile + "/" + "f*.json"))
    output_list = []

    for f in read_files:
        with open(f, "rb") as infile:
            output_list.append(json.load(infile))

    print 'creating the combined json FINAL file'
    with open("./" + outputfile + "/" + "FINAL.json", "wb") as outfile:
        json.dump(output_list, outfile)


