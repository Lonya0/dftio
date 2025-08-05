import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dftio import __version__
from dftio.io.parse import ParserRegister
from tqdm import tqdm
from multiprocessing.pool import Pool
from dftio.logger import set_log_handles
from dftio.plot.plot_eigs import BandPlot

def get_ll(log_level: str) -> int:
    """Convert string to python logging level.

    Parameters
    ----------
    log_level : str
        allowed input values are: DEBUG, INFO, WARNING, ERROR, 3, 2, 1, 0

    Returns
    -------
    int
        one of python logging module log levels - 10, 20, 30 or 40
    """
    if log_level.isdigit():
        int_level = (4 - int(log_level)) * 10
    else:
        int_level = getattr(logging, log_level)

    return int_level

def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="dftio is to assist machine learning communities to transcript DFT output into a format that is easy to read or used by machine learning models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-v', '--version', 
                        action='version', version=f'%(prog)s {__version__}', help="show the dftio's version number and exit")


    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    # log parser
    parser_log = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser_log.add_argument(
        "-ll",
        "--log-level",
        choices=["DEBUG", "3", "INFO", "2", "WARNING", "1", "ERROR", "0"],
        default="INFO",
        help="set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO "
             "and 3=DEBUG",
    )

    parser_log.add_argument(
        "-lp",
        "--log-path",
        type=str,
        default=None,
        help="set log file to log messages to disk, if not specified, the logs will "
             "only be output to console",
    )

    # config parser
    parser_parse = subparsers.add_parser(
        "parse",
        parents=[parser_log],
        help="parse dataset from DFT output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser_parse.add_argument(
        "-m",
        "--mode",
        type=str,
        default="abacus",
        help="The name of the DFT software, currently support abacus/rescu/siesta/gaussian/pyatb",
    )

    parser_parse.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used to parse the dataset. (For n>1, we use the multiprocessing to accelerate io.)",
    )

    parser_parse.add_argument(
        "-r",
        "--root",
        type=str,
        default="./",
        help="The root directory of the DFT files.",
    )

    parser_parse.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="frame",
        help="The prefix of the DFT files under root.",
    )

    parser_parse.add_argument(
        "-o",
        "--outroot",
        type=str,
        default="./",
        help="The output root directory.",
    )

    parser_parse.add_argument(
        "-f",
        "--format",
        type=str,
        default="dat",
        help="The output file format, should be dat, ase or lmdb.",
    )

    parser_parse.add_argument(
        "-ham",
        "--hamiltonian",
        action="store_true",
        help="Whether to parse the Hamiltonian matrix.",
    )

    parser_parse.add_argument(
        "-ovp",
        "--overlap",
        action="store_true",
        help="Whether to parse the Overlap matrix",
    )
    parser_parse.add_argument(
        "-dm",
        "--density_matrix",
        action="store_true",
        help="Whether to parse the Density matrix",
    )

    parser_parse.add_argument(
        "-eig",
        "--eigenvalue",
        action="store_true",
        help="Whether to parse the kpoints and eigenvalues",
    )
    parser_parse.add_argument(
        "-min",
        "--band_index_min",
        type=int,
        default=0,
        help="The initial band index for eigenvalues to save.(0-band_index_min) bands will be ignored!"
    )
    
    parser_band = subparsers.add_parser(
        "band",
        parents=[parser_log],
        help="plot band for eigenvalues data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_band.add_argument(
        "-r",
        "--root",
        type=str,
        default="./",
        help="The root directory of eigenvalues data.",
    )
    parser_band.add_argument(
        "-f",
        "--format",
        type=str,
        default=None,
        help="load file format, should be dat or ase. default is None, which means auto detect.",
    )
    
    parser_band.add_argument(
        "-min",
        "--band_index_min",
        type=int,
        default=0,
        help="The minimum band index to plot.",
    )
    parser_band.add_argument(
        "-max",
        "--band_index_max",
        type=int,
        default=None,
        help="The maximum band index to plot.",
    )

    return parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments and convert argument strings to objects.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv

    Returns
    -------
    argparse.Namespace
        the populated namespace
    """
    parser = main_parser()
    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()
    else:
        parsed_args.log_level = get_ll(parsed_args.log_level)

    return parsed_args

class wapper:
    def __init__(self, args):
        self.args = args
        self.parser = ParserRegister(
            **self.args
        )
        
    def __call__(self, idx):
        # print(idx)
        self.parser.write(idx=idx, **self.args)
        return True

def main():
    args = parse_args()

    if args.command not in (None, "train", "test", "run"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "parse":
        parser = ParserRegister(
                        **dict_args
                    )
        
        if args.num_workers > 1:
            with Pool(args.num_workers) as p:
                list(tqdm(p.imap(wapper(dict_args), range(len(parser))), total=len(parser), desc="Parsing the DFT files: "))
        else:
            for i in tqdm(range(len(parser)), desc="Parsing the DFT files: "):
                parser.write(idx=i, **dict_args)
        
    if args.command == "band":
        bandplot = BandPlot(
            **dict_args
        )
        bandplot.load_dat(fmt=args.format)
        bandplot.plot(bmin=args.band_index_min, bmax=args.band_index_max)

if __name__ == "__main__":
    main()