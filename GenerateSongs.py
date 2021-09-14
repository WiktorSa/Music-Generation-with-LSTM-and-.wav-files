import argparse
from generatemusic import generate_music


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', default='seeds', help='Directory with songs used as seeds', type=str, required=False)
    parser.add_argument('-md', default='model_weights', help='Directory with saved model weights', type=str,
                        required=False)
    parser.add_argument('-nm', default='normalizer', help='Directory with saved normalizer', type=str, required=False)
    parser.add_argument('-ad', default='generated_songs', help='Directory in which generated songs should be saved',
                        type=str, required=False)
    parser.add_argument('-ss', default=5, help='The second from which the function should take seed', type=int,
                        required=False)
    parser.add_argument('-lp', default=10, help='Length of generated piece (in seconds)', type=int, required=False)
    parser.add_argument('-sr', default=16000, help='Sample rate of all songs (in Hz)', type=int, required=False)
    parser.add_argument('-ls', default=0.25, help='Length of sample used in training (in seconds)', type=float,
                        required=False)
    parser.add_argument('-hs', default=2048, help='Hidden size used in training', type=int, required=False)
    parser.add_argument('-dp', default=0.2, help='Value of dropout (linear layer) used in training', type=float,
                        required=False)
    return parser


if __name__ == '__main__':
    data_parser = create_parser()
    args = data_parser.parse_args()
    generate_music(args.sd, args.md, args.nm, args.ad, args.ss, args.lp, args.sr, int(args.sr * args.ls * 2),
                  args.hs, int(args.sr * args.ls), args.dp)
