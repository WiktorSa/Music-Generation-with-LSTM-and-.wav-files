import argparse
from preprocessing import generate_and_save_data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', help='Location of files with raw audio', type=str, required=True)
    parser.add_argument('-sr', default=16000, help='Sample rate of all songs (in Hz)', type=int, required=False)
    parser.add_argument('-sf', default=4, help='Sample frequency (in Hz)', type=int, required=False)
    parser.add_argument('-lw', default=4000, help='The length of window', type=int, required=False)
    parser.add_argument('-lp', default=10, help='Length of piece (in seconds)', type=int, required=False)
    parser.add_argument('-ts', default=0.8, help='Size of trial data', type=float, required=False)
    parser.add_argument('-vs', default=0.1, help='Size of validation data', type=int, required=False)
    parser.add_argument('-df', default='data', help='Folder where data should be stored', type=str, required=False)
    parser.add_argument('-sd', default=1001, help='Seed', type=int, required=False)
    return parser


if __name__ == '__main__':
    data_parser = create_parser()
    args = data_parser.parse_args()
    generate_and_save_data(args.lf, args.sr, args.sf, args.lw, args.lp, args.ts, args.vs, args.df, args.sd)
