import tarfile
import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--fname", default="./archive.tar.gz", type=str)
args = parser.parse_args()
print(args) 
print()

def main(fname):
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()

if __name__ == "__main__":
    main(args.fname)