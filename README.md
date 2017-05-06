# RankLyrics
automatic song lyrics generator


This system is currently trained to generate lyrics using a corpus of Rock song lyrics. For comprehensive documentation see this accompanying short paper: http://zangsir.weebly.com/uploads/3/1/3/8/3138983/ranklyrics.pdf

Two examples are included below, more examples: https://github.com/zangsir/songLyrics/blob/master/txt/samples.txt

## Dependency
The system runs on Python 3. It depends on a number of libraries and models:

- kenlm (python module https://github.com/kpu/kenlm#python-module)
- gensim (https://radimrehurek.com/gensim/)
- SRILM(binary included, no need to do anything)
- spaCy (https://spacy.io/)
- word2vec Google News vectors (https://github.com/mmihaltz/word2vec-GoogleNews-vectors), should be placed under the model/ directory

## Usage
### Commandline Usage
There are two modes: a Fast and a Slow mode (refer to the paper linked above), with a tradeoff between speed and interestingness of lyrics. The slow mode often generates more interesting lyrics with more variations in structure.But it is 8x slower than Fast mode.

To generate lyrics, simply do:

<code>python songLyrics.py N mode</code>

where N is the number of songs you want to generate lyrics, and mode=1 for Slow mode and mode=0 for Fast mode. For instance, to generate 5 songs in Fast mode:

<code>python songLyrics.py 5 0 </code>

You will find the resulting lyrics in <code>lyrics_gen.txt</code>. 

### Python programming usage
```python
# this import might take a minute due to loading the GoogleNewsVector model(3.6G)
from modules.lyrics_classes import SongLyrics
# default is generating using Fast mode, for Slow mode, use online=True argument
lyrics = SongLyrics()
verses = lyrics.gen_verses(num_of_lines)
```

where num_of_lines is the desired integer number of lyrics lines you want to generate. 

Note that loading the Google News vector (3.5G) can take more than 1 minute in the beginning, and in the slow mode, it can take up to 3 minutes to generate one song. (please refer to the paper for explanations)




## Example output (more: songLyrics/txt/samples.txt)
(1)

Verses:

pain disappear, i'll do anything.

I have everything

I feel you near

You look at me

We're gonna save me?

Why can't you see?

Making me 


Chorus:


Poisonous lookalike

Voodoo, voodoo

O girlfriend

I've Lord

Radicals

(2)

Verses:

All these little victories, yeah

Be my woman, yeah.

I like to Oh yeah

Oh yeah,

Let me go

Oh no


Chorus:

Sad song, sad song

Don't know where I belong

Do-be-da, do-be-da

Like a no remorse.

There's no return
