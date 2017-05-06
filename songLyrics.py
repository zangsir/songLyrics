from modules.lyrics_classes import SongLyrics
from modules.feature_ext_utils import *
import sys
from sklearn.externals import joblib
import random

def gen_verse_part(lyrics, num_verse):
    verses = lyrics.gen_verses(num_verse)
    return '\n'.join(verses)


def gen_chorus_part(lyrics, num_chorus):
    chorus = lyrics.gen_verses(num_chorus, False)
    return '\n'.join(chorus)


def gen_song(verses, choruses):
    return '\n===========\nVerses:\n%s \n\nChorus:\n%s' % (verses, choruses)


def serialize_song(song, outputfile):
    f = open(outputfile, 'a')
    f.write(song + '\n\n++++++++++++++\n')


def main():
    num_songs = sys.argv[1]
    online = int(sys.argv[2])
    num_verse = random.randint(5,11)
    num_chorus = 4
    lyrics = SongLyrics(online=online)
    print('finished loading models...')
    for i in range(int(num_songs)):
        print(i)
        verses = gen_verse_part(lyrics, num_verse)
        choruses = gen_chorus_part(lyrics, num_chorus)
        song = gen_song(verses, choruses)
        serialize_song(song, 'lyrics_gen.txt')


if __name__ == '__main__':
    main()
