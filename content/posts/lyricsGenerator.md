---
title: "Can a language model rap like Eminem?"
date: 2023-02-18T12:38:33+01:00
toc: false
images:
tags:
  - language model
  - nlp
  - markov chain
  - eminem
  - lyrics
---

## Lyrics Generator using Markov Chains

Nowadays, natural language processing (NLP), especially language models, are all around us. Without learning language or defining fixed rules it is pretty amazing how well they can generate new content also including art. This is a hotly debated topic [Towards Science: AI Art Debate](https://towardsdatascience.com/the-ai-art-debate-excitement-fear-and-ethics-c04d30f338da). Besides the ethical discussion, language models open up a large field of use cases. For example in combination with data sources containing lyrics, songs could be generated. The aim of this project work is to analyze songs from one specific artist and try to generate more lyrics which could be sold as new songs. Not only the simple generation would be interesting, but also how to generate ryhmes by using an Markov Chains.

## Authentication
To get correct lyrics in an consistent format, the [Genius API](https://docs.genius.com/) is used. Therefore a oAuth process is required. In the Genius API, an client needs to be created in order to get an access token for authentication. By using the python package [Lyricsgenius](https://lyricsgenius.readthedocs.io/en/master/) all endpoints concerning artists and songs can be called easily inside of the code.


```python
import lyricsgenius
```


```python
ACCESS_TOKEN = ''
ARTIST = "Eminem"
```


```python
geniusAPI = lyricsgenius.Genius(ACCESS_TOKEN)
```

## Collecting data

First of all, we check whether the artist is available in the API. In respect to the response time from the server, we need to search the song list first and do an individual api call for each song lyrics.


```python
artist = geniusAPI.search_artist(ARTIST, max_songs=1)
artist
```

    Searching for songs by Eminem...
    
    Song 1: "Rap God"
    
    Reached user-specified song limit (1).
    Done. Found 1 songs.
    




    Artist(id, songs, ...)




```python
songList = geniusAPI.artist_songs(artist.id, per_page=50)
```


```python
songs = []
for song in songList['songs']:
    if song['lyrics_state'] == 'complete': # check if the lyrics is available
        songs.append(geniusAPI.lyrics(song['id']))
```

    Couldn't find the lyrics section. Please report this if the song has lyrics.
    Song URL: https://genius.com/Shabaam-sahdeeq-5-star-generals-instrumental-lyrics
    Couldn't find the lyrics section. Please report this if the song has lyrics.
    Song URL: https://genius.com/Eminem-8-mile-instrumental-lyrics
    Couldn't find the lyrics section. Please report this if the song has lyrics.
    Song URL: https://genius.com/Jay-z-8-miles-and-runnin-instrumental-lyrics
    

In order to get some randomness into the data, shuffle is used.


```python
import random
```


```python
random.shuffle(songs)
```

## Preprocessing
In this section, all unnecessary information is removed to get the pure lyrics string. While working with language models, most of the time line breaks are removed. But for rhymes those breaks are important. That is why we leave them in the text.


```python
import re
```


```python
songs[0]
```




    '2nd Round Freestyle Lyrics[Intro]\nClinton Sparks\n("Obie Trice" "Yeah")\nAnger Management 3 (Get familiar, that\'s right)\nShady\nHahaha (Shoutout to Ireland\nScotland, London) Obie\u2005Trice\nSecond\u2005Rounds On Me\u2005(My man Tim Westwood) Hahaha (DJ\u2005Semtex)\n(Real DJ, bitches) New album coming this summer\nI know you been waitin\' ([?])\n[Verse]\nYeah\nI’m back to business, back to them classic hits\nMacking bitches in the massive whips\nSubtracting other rappers chips, \'cause I’m actually\nWhat’s happening, these other cats talent is shit\nI’m talented as shit, bringing gallons with me, man, I’m so sick\nIt’s like a challenge every time that I spit\nSo how do you balance these amateurs with a graduate\n[?], no candidate could ever exist\nThis ain\'t Canada, ni-nigga, this \'caine carrying\nHand on a fifth, came in the game wearing the same fit\nMy funny shit, you niggas took it extreme there (Yeah)\nYou still catch a couple of clips up in your Spring wear\nOn the scuffle tip [?]You might also likeEmbed'



After checking the first element in the lyrics list, we can see that ther is a first section which contains title information. Important section inside of the song such as Intro and Chorus are marked with brackets. In addition to that, at the end there is always a number in combination with the word "Embeded". Such information is not useful for the language model. That is why I am removing such content in the preprocessing step using regex.


```python
def cleaning(lyrics) -> str:
    # first translation information
    lyrics = re.sub(r"^[^_]* Lyrics", "",lyrics)
    # song sections
    lyrics = re.sub(r"(\[.*?\])", "", lyrics)
    # Embed information at the end 
    return re.sub(r"((\d.?\dK?)?Embed)", "", lyrics)
```


```python
cleaned_lyrics = [cleaning(lyric) for lyric in songs if type(lyric) == str]
```


```python
print("Available cleaned songs:", len(cleaned_lyrics))
```

    Available cleaned songs: 44
    

## Model generation
After the data cleaning is completed, the input text is ready to be processed by the model.


```python
import uuid
```

To be able to analyze the generated text afterwards, an unique id is used to save input and output text.


```python
uuid_song = str(uuid.uuid4())
uuid_song
```




    '41476e3a-df2d-47b1-ae9e-4ee1b9b7870f'



By joining all song lyrics, we get the basic corpus for the Markov Chain. See the next section for a detailed description.


```python
input_text = ' '.join(cleaned_lyrics)
file = open('input_text/' + uuid_song + ".txt", "w")
file.write(input_text)
file.close()
```

### Markov Chains
Markov chains are useful mathematical models that use concepts from probability and matrix algebra to generate text. While training the Markov Chain, a matrix is generated which calculates the probability of the next word or character based on the previous used text. \
See: [An Introduction to Markov Chains](http://dx.doi.org/10.13140/2.1.1833.8248)

#### Word-based generation vs Character-based generation
There are two possible ways to create an Markov Chain. For word-based models, the probability of the next word is calculated. While in the character-based approach each character is weighted individually. \
One side effect of word-based generation is that the vocabulary only includes words which are already known. This can be tricky for lyrics generation, because sometimes for ryhme purposes a new word can be created or a word from another language can be used. Even though I don't think that this model is able to create new words, it would be better to use the character-based approach.
Another reason why I choose a character-based models are the line breaks. While looking at a song text, one can see that the lines define the rhyme. This generation can be achieved easier by generating characters.


```python
def getTransitionTable(data, k = 4):#if X is the sequence of 'k = 3' and Y is predicted character or k+1 the character
    T = {} #making an empty dictionary
    
    for i in range(len(data) - k):
        X = data[i:i+k]
        Y = data[i+k]
# making dictornary for each after word and new that are not in dict of x(transition dict)
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None: #checking is y is not present or notin Transition Dictonary(x)
                T[X][Y] = 1
            else:
                T[X][Y] +=1
    
    return T
```

The transition table helps us to get an overview over all available characters in the dictionary and their current frequency in the input text (corpus). The variable *k* defines the number of characters which are considered for the selection of the next character. That is why this variable defines the dimension of our transition table.


```python
lyrics_transition_table = getTransitionTable(input_text, k = 4)
# show first 50 items
list(lyrics_transition_table.items())[0:10]
```




    [('\nCli', {'n': 2, 'm': 1, 'p': 1}),
     ('Clin', {'t': 3}),
     ('lint', {'o': 3}),
     ('into', {'n': 3, ' ': 22}),
     ('nton', {' ': 2, '\n': 1}),
     ('ton ', {'S': 1, 's': 3, 't': 1}),
     ('on S', {'p': 1}),
     ('n Sp', {'a': 2}),
     (' Spa', {'r': 1, 'n': 1}),
     ('Spar', {'k': 1})]




```python
def convertFreqIntoProb(T):
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
            
    return T
```

In order to get a probability for the each character or character group, the frequency is used.


```python
char_model = convertFreqIntoProb(lyrics_transition_table)
list(char_model.items())[0:10]
```




    [('\nCli', {'n': 0.5, 'm': 0.25, 'p': 0.25}),
     ('Clin', {'t': 1.0}),
     ('lint', {'o': 1.0}),
     ('into', {'n': 0.12, ' ': 0.88}),
     ('nton', {' ': 0.6666666666666666, '\n': 0.3333333333333333}),
     ('ton ', {'S': 0.2, 's': 0.6, 't': 0.2}),
     ('on S', {'p': 1.0}),
     ('n Sp', {'a': 1.0}),
     (' Spa', {'r': 0.5, 'n': 0.5}),
     ('Spar', {'k': 1.0})]



## Lyrics generation
By using this probability table, our new lyrics can be generated.


```python
import numpy as np
```


```python
def sample_next(context, T, k = 4):
    context = context[-k:] #AS WE are predict next char from last k char 
    
    
    if T.get(context) is None:
        return ' '
    
    possible_chars = list(T[context].keys())
    possible_probabs = list(T[context].values())
    
    return np.random.choice(possible_chars, p =possible_probabs )
```


```python
def generateText(starting_sent,T, k = 4, max_len = 100):
    sentence = starting_sent
    
    context = sentence[-k:]
    
    for i in range(max_len):
        next_pred = sample_next(context, T, k)
        sentence += next_pred
        context = sentence[-k:]
        
    return sentence
```

By check the mean lyrics length of Eminem songs we can find our new lyrics max length.


```python
lens = [len(lyric) for lyric in cleaned_lyrics]
np.mean(lens)
```




    3269.0454545454545



For the text generation, we leave our *k* with 4. This means that we need at least 4 input characters to be able to generate new content.


```python
lyrics_predict = generateText("Look ", char_model, max_len=4000, k=4)
```


```python
file = open('output_text/' + uuid_song + ".txt", "w")
file.write(lyrics_predict)
file.close()
print(lyrics_predict)
```

    Look me -- fuck a Rugers
    If yo' questin' do'
    But my side, twenty-five it
    The potions:
    13. "Wanna man Probably real he sight one vexed
    That's like this shit, like scratchin' yet, you don't bitch
    Keep asking to camera
    Parket
    These same pedes
    I'm 'bout of serious when I was that or all Street; I wild number off right hole
    I got the bullshit? (No!)
    We polyps'll shot
    And the hey, we does onto more top us on my style for the fucking a miss you ain't was on
    I call my course same beat shit up on most no car like Georget points jerked eyebrown
    Only what big eyebrows
    You might also like 
    I used to my rid on, ni-niggas when take it)
    Pour you count checks
    Till me to hypnotic, it's after pull of my venomous human the 'jectable plane
    If he stomach othere's got
    But I see cold, try, Marticle, to this soul
    Welcome words
    F around an odds watch never, dawg
    Freestyle
    
    Yo, but your your trave?
    Yours
    Prete giving (Yo)
    The Chewbaca's no one and we resort a gang of chroneral
    There's Greg with album come sent crunk it and up now, so cover existen the won't underine people tip Youngest of the Jackin' lyrics I message it start they ain't having but keep as I get wasn't fitter 'cause the push
    Us and sinners
    How when I lookin' a pitbullshit to back therfuckin' do? (Yeah I never gave your plan
    Tiki
    Girl, we he wear too class off a (Dahmer, a make hit some Big Pun
    Wanna punk
    You sound, drink
    Send avoid land, I’m all thout (Guzzles, bullshit, why I gets dull to just hole
    Now, we get mattent, but pickets release
    Dethrough, I puts your so love you scramblers do your changer MC from the reach fucker represent name
    And it's okay, tryna bunched up, and runnin' that's one in the marriage
    The road dead
    The mother a fake-ass it something
    And I candle, that I go?
    When chrome addities layin' gats (*slap his shit to was shit that make hired that the puts form of pile Road
    You might, you can let's talkin
    She game, hate alongers
    Dignital
    Yeah I got coming out (Guzzle it, Devil
    Innovatively me
    'Cause I'ma brain, I'll get class
    All you say in a sweet niggas, I'll he ain't shit
    You wanna her hands with the pure I would number her the sundial, richest fetuses to usin' and the mass over" by drunk, your faces laying with concreaming of an audience to doin'?) Man, I'll her you
    If you don't know hard
    Burnin', I tryin', money said they head make ya robot, factors, shoes, don't handle it)
    Pour 40 out my freestyles, Pt IITime fireplyin' next, gotta go
    To this fucking you can't know the flow do my business
    And your genius, top a split
    Cutting that nigga fricance you want start rapperson, don’t someone who beast sticuffleupagus
    But it the Westwood!)
    We womens' feeding at ya ass rapped to cuss a worth my own and lived
    I said you action
    Where in this a needs, bit, even on thrown a Cheddar's to battlin' Puerto Rick came to Stat Quo
    Still in you better what eye so I'ma make it up date like still you a dozen tryna check that I go out it brothere I can seen Lawrence, or for ya run
    And traight might, list
    High-flyin'
    For spleen go any did
    Forty hot
    And that's goin' having
    Y'all no four-four Pac, on sometime twenty grown
    Off the day, the hard to give fool like the Lyrics are
    Me and Ms. Cleave ear, then
    With with Canada, niggas feel Me FlowPlayin' do?
    I'll shit mess
    Emine's of the walk the stomaching at you this big brows
    
    Walkin' with he glass are book; fucket
    It put you before than Corvetter, we in vain
    They don’t like zeros
    'Cause a far this suin' ever seen it and in her man take it also like, Tyrone, casualty to the busin
    I'm just kill follow)
    Only hot like
    Take hits for nostrack
    (I'm noddities her block, one people the playin' lyric versation
    I’m fuck median woments them hot suin' drugs with they was halfway this bitch up out, how if I was you before..."
    "Rhymes this my malt
    This near (Uh, bitch, and fly night be that yo you said, call past since I layin', them ho's in the flow
    At the green heave me boy she colla
    Fuckin' yo we know heir Jay Delta"
    That'll the to give remember
    December the sta
    

## Conclusion and Further work
Markov Chains are a great way to generate text language independent. Two big advantages by using this method are that there is no need for large input data and that the compution time for the model is very short. I think it is quite amazing how fast a language model can learn about sentences and grammar. \
By looking at our results, the model was able to use line breaks which structures our generated lyrics very well. Nevertheless, there is lots of room for improvement. One can hardly find ryhmes in the text and there is no context in the song text. \
One approach would be to change the hyperparamter *k* to get more related text lines. It is questionable whether by using larger character groups already existing phrases of songs will be reused as a result. Another idea would be to investigate more time in the preprocessing. One could analyze the ryhmes of the existing songs and select only unison rhymes as input text. I think that double rhymes or cross rhymes confuses the model because the connection of the text lines are too far apart. \
To sum up, I am happy with the result and I think that with more preprocessing or hyperparameter tuning a language model could rap like Eminem. Even if it takes a lot more than rhymes to create new songs.

GitHub Repo: [Lyrics Generator](https://github.com/lauragregorc/lyrics-generator)

References: 
- [https://github.com/soniajoseph/MarkovLyric](https://github.com/soniajoseph/MarkovLyric) 
- [https://lyricsgenius.readthedocs.io](https://lyricsgenius.readthedocs.io) 
- [https://docs.genius.com/](https://docs.genius.com/) 
- [https://github.com/aryangulati/Character-Based-Language-Model](https://github.com/aryangulati/Character-Based-Language-Model) 
- [An Introduction to Markov Chains](http://dx.doi.org/10.13140/2.1.1833.8248)
