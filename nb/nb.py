import os
import sys


"""

Command line tools for nbnb neural builder

"""
import os
import sys
import argparse
from colorama import Fore, Back, Style
import traceback
from .version import __version__, short_version
from datetime import datetime

dt = datetime.now()


__VERSION__ = f"👍    {__version__}"
__AUTHOR__ = "😀    Lucas Jin"
__CONTACT__ = "😍    telegram: lucasjin"
__DATE__ = f"👉    {dt.strftime('%Y.%m.%d')}, since 2020.11.11"
__LOC__ = "👉    Shenzhen, China"
__git__ = "👍    http://github.com/jinfagang/nb"


def arg_parse():
    """
    parse arguments
    :return:
    """
    parser = argparse.ArgumentParser(prog="alfred")
    parser.add_argument(
        "--version", "-v", action="store_true", help="show version info."
    )

    # vision, text, scrap
    main_sub_parser = parser.add_subparsers()

    # =============== vision part ================
    vision_parser = main_sub_parser.add_parser(
        "vision", help="vision related commands."
    )
    vision_sub_parser = vision_parser.add_subparsers()

    vision_extract_parser = vision_sub_parser.add_parser(
        "extract", help="extract image from video: alfred vision " "extract -v tt.mp4"
    )
    vision_extract_parser.set_defaults(which="vision-extract")
    vision_extract_parser.add_argument("--video", "-v", help="video to extract")
    vision_extract_parser.add_argument(
        "--jumps", "-j", help="jump frames for wide extract"
    )

    # =============== text part ================
    text_parser = main_sub_parser.add_parser("text", help="text related commands.")
    text_sub_parser = text_parser.add_subparsers()

    text_clean_parser = text_sub_parser.add_parser("clean", help="clean text.")
    text_clean_parser.set_defaults(which="text-clean")
    text_clean_parser.add_argument("--file", "-f", help="file to clean")

    text_translate_parser = text_sub_parser.add_parser("translate", help="translate")
    text_translate_parser.set_defaults(which="text-translate")
    text_translate_parser.add_argument(
        "--file", "-f", help="translate a words to target language"
    )
    return parser.parse_args()


def print_welcome_msg():
    print("-" * 70)
    print(
        Fore.BLUE
        + Style.BRIGHT
        + "              NB. "
        + Style.RESET_ALL
        + Fore.WHITE
        + "- Valet of Artificial Intelligence."
        + Style.RESET_ALL
    )
    print(
        "         Author : " + Fore.CYAN + Style.BRIGHT + __AUTHOR__ + Style.RESET_ALL
    )
    print(
        "         Contact: " + Fore.BLUE + Style.BRIGHT + __CONTACT__ + Style.RESET_ALL
    )
    print(
        "         At     : "
        + Fore.LIGHTGREEN_EX
        + Style.BRIGHT
        + __DATE__
        + Style.RESET_ALL
    )
    print(
        "         Loc    : "
        + Fore.LIGHTMAGENTA_EX
        + Style.BRIGHT
        + __LOC__
        + Style.RESET_ALL
    )
    print(
        "         Star   : " + Fore.MAGENTA + Style.BRIGHT + __git__ + Style.RESET_ALL
    )
    print(
        "         Ver.   : " + Fore.GREEN + Style.BRIGHT + __VERSION__ + Style.RESET_ALL
    )
    print("-" * 70)
    print("\n")


def main(args=None):
    args = arg_parse()
    if args.version:
        print(print_welcome_msg())
        exit(0)
    else:
        args_dict = vars(args)
        print_welcome_msg()
        try:
            module = args_dict["which"].split("-")[0]
            action = args_dict["which"].split("-")[1]
            print(Fore.GREEN + Style.BRIGHT)
            print(
                "=> Module: "
                + Fore.WHITE
                + Style.BRIGHT
                + module
                + Fore.GREEN
                + Style.BRIGHT
            )
            print("=> Action: " + Fore.WHITE + Style.BRIGHT + action)
            if module == "vision":
                if action == "extract":
                    pass
                elif action == "reduce":
                    pass

            elif module == "text":
                if action == "clean":
                    f = args_dict["file"]
                    print(Fore.BLUE + Style.BRIGHT + "Cleaning from {}".format(f))
                elif action == "translate":
                    f = args.v
                    print(Fore.BLUE + Style.BRIGHT + "Translate from {}".format(f))
        except Exception as e:
            traceback.print_exc()
            print(Fore.RED, "parse args error, type -h to see help. msg: {}".format(e))


if __name__ == "__main__":
    main()
