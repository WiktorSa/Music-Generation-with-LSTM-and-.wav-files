import argparse
from preprocessing import generate_and_save_data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', default='raw_audio', help='Directory with raw audio', type=str, required=False)
    parser.add_argument('-sr', default=16000, help='Sample rate of all songs (in Hz)', type=int, required=False)
    parser.add_argument('-ls', default=120, help='Length of song (in seconds)', type=int, required=False)
    parser.add_argument('-lp', default=0.25, help='Length of sample (in seconds)', type=float, required=False)
    parser.add_argument('-ts', default=0.6, help='Size of trial data', type=float, required=False)
    parser.add_argument('-vs', default=0.2, help='Size of validation data', type=int, required=False)
    parser.add_argument('-sf', default='data', help='Folder where data should be stored', type=str, required=False)
    parser.add_argument('-sn', default='normalizer', help='Folder where normalizers should be stored',
                        type=str, required=False)
    parser.add_argument('-sd', default=1001, help='Seed', type=int, required=False)
    return parser


if __name__ == '__main__':
    data_parser = create_parser()
    args = data_parser.parse_args()
    generate_and_save_data(args.dr, args.sr, args.ls, args.lp, args.ts, args.vs, args.sf, args.sn, args.sd)
