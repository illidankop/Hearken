import os

import csv

# PATH_TO_ROOT = os.path.join(
#     r"C:\Users\Public\Documents\Classification\PANN_files")


# with open("/home/daniel/elta_projects/Hearken/models_ds/class_labels_indices.csv", 'r') as f:
#     reader = csv.reader(f, delimiter=',')
#     lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)
classes_num = len(labels)


motors_sounds_1 = {
    'Vehicle', 'Sailboat, sailing ship', 'Rowboat, canoe, kayak',
    'Motorboat, speedboat', 'Ship', 'Motor vehicle (road)', 'Car', 'Car passing by',
    'Race car, auto racing', 'Truck', 'Bus', 'Emergency vehicle', 'Motorcycle',
    'Train', 'Train whistle', 'Aircraft', 'Aircraft engine', 'Jet engine', 'Propeller, airscrew',
    'Helicopter', 'Fixed-wing aircraft, airplane', 'Engine', 'Light engine (high frequency)',
    "Dental drill, dentist's drill", 'Lawn mower', 'Chainsaw', 'Medium engine (mid frequency)',
    'Heavy engine (low frequency)', 'Engine knocking', 'Engine starting', 'Idling',
    'Accelerating, revving, vroom', 'Microwave oven', 'Blender', 'Hair dryer',
    'Electric toothbrush', 'Vacuum cleaner', 'Electric shaver, electric razor', 'Jackhammer',
    'Power tool', 'Drill',
}

motors_sounds_2 = {'Hum', 'Throbbing', 'Buzz', "Mains hum", "Steam whistle", "Train horn", "Hiss", "Foghorn", "Mechanical fan", "Air conditioning", "Humming","Mechanisms"}

motors_sounds = motors_sounds_1 | motors_sounds_2 

music_sounds_1 = {
    'Music', 'Guitar', 'Musical instrument', 'Drum', 'Drum kit', 'Bass drum', 
    'Percussion', 'Snare drum', 'Cymbal', 'Drum machine', 'Rimshot', 'Drum roll', 'Scary music'
    
}
music_sounds_2 = {'Plucked string instrument', 'Electric guitar', 'Bass guitar', 'Acoustic guitar', 
                    'Steel guitar, slide guitar', 'Tapping (guitar technique)', 'Strum', 'Banjo', 'Sitar',
                    'Mandolin', 'Zither', 'Ukulele', 'Keyboard (musical)', 'Piano', 'Electric piano', 'Organ', 
                    'Electronic organ', 'Hammond organ', 'Synthesizer', 'Sampler', 'Harpsichord', 'Timpani', 
                    'Tabla', 'Hi-hat', 'Wood block', 'Tambourine', 'Rattle (instrument)', 'Maraca', 'Gong', 
                    'Tubular bells', 'Mallet percussion', 'Marimba, xylophone', 'Glockenspiel', 'Vibraphone', 
                    'Steelpan', 'Orchestra', 'Brass instrument', 'French horn', 'Trumpet', 'Trombone', 
                    'Bowed string instrument', 'String section', 'Violin, fiddle', 'Pizzicato', 'Cello', 'Double bass',
                    'Wind instrument, woodwind instrument', 'Flute', 'Saxophone', 'Clarinet', 'Harp', 'Jingle bell', 
                    'Bicycle bell', 'Tuning fork', 'Chime', 'Wind chime', 'Change ringing (campanology)', 
                    'Harmonica', 'Accordion', 'Bagpipes', 'Didgeridoo', 'Shofar', 'Theremin', 'Singing bowl', 
                    'Scratching (performance technique)', 'Pop music', 'Hip hop music', 'Beatboxing', 'Rock music', 
                    'Heavy metal', 'Punk rock', 'Grunge', 'Progressive rock', 'Rock and roll', 'Psychedelic rock', 
                    'Rhythm and blues', 'Soul music', 'Reggae', 'Country', 'Swing music', 'Bluegrass', 'Funk', 
                    'Folk music', 'Middle Eastern music', 'Jazz', 'Disco', 'Classical music', 'Opera', 'Electronic music',
                    'House music', 'Techno', 'Dubstep', 'Drum and bass', 'Electronica', 'Electronic dance music', 'Ambient music', 
                    'Trance music', 'Music of Latin America', 'Salsa music', 'Flamenco', 'Blues', 'Music for children',
                    'New-age music', 'Vocal music', 'A capella', 'Music of Africa', 'Afrobeat', 'Christian music', 'Gospel music', 
                    'Music of Asia', 'Carnatic music', 'Music of Bollywood', 'Ska', 'Traditional music', 'Independent music', 'Song', 
                    'Background music', 'Theme music', 'Jingle (music)', 'Soundtrack music', 'Lullaby', 'Video game music', 'Christmas music', 
                    'Dance music', 'Wedding music', 'Happy music', 'Funny music', 'Sad music', 'Tender music', 'Exciting music', 
                    'Angry music',}

music_sounds = music_sounds_1 | music_sounds_2 

silence_sounds ={'Silence'}

noise_sounds = { 'Sine wave', 'Harmonic', 'Chirp tone', 'Sound effect', 'Pulse', 'Inside, large room or hall', 'Inside, public space', 
                 'Reverberation', 'Echo', 'Noise', 'Environmental noise', 'Static', 'Distortion', 'Sidetone', 'Cacophony','Pink noise',
                 'Vibration', 'Television', 'Radio', 'Field recording' , 'Electronic tuner', 'Effects unit', 'Chorus effect', 'White noise'}

human_sounds_1 = {
    'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking', 
    'Conversation', 'Narration, monologue', 'Speech synthesizer', 'Shout', 'Yell', 
    'Children shouting', 'Screaming', 'Whispering', 'Laughter', 'Baby laughter', 'Giggle', 
    'Belly laugh', 'Chuckle, chortle', 'Crying, sobbing', 'Baby cry, infant cry', 'Sigh', 'Singing', 
    'Groan', 'Grunt', 'Breathing', 'Wheeze', 'Snoring', 'Whistle','Hubbub, speech noise, speech babble', 
    'Children playing', 'Growling', 'Walk, footsteps', 'Speech'
}

human_sounds_2 = {'Babbling', 'Whoop', 'Male singing', 'Female singing', 'Child singing', 'Synthetic singing', 'Rapping', 
                  'Whistling', 'Gasp', 'Pant', 'Snort', 'Cough', 'Throat clearing', 'Sneeze', 'Sniff', 'Cheering', 'Applause', 
                  'Chatter', 'Crowd', 'Yip', 'Bow-wow', 'Whimper (dog)', 'Caterwaul', 'Battle cry', 
                  'Snicker', 'Whimper', 'Choir', 'Yodeling', 'Chant', 'Mantra', 'Run', 'Shuffle', 'Chewing, mastication', 'Biting', 
                  'Gargling', 'Stomach rumble', 'Burping, eructation', 'Hiccup', 'Fart', 'Hands', 'Finger snapping', 'Clapping', 
                  'Heart sounds, heartbeat', 'Heart murmur'} #'Clucking', 'Pattering', 

human_sounds = human_sounds_1 | human_sounds_2

inhouse_sounds = {
    'Inside, small room', 'Water tap, faucet'
}



indoor_sounds = {'Alarm', 'Alarm clock', 'Bathtub (filling or washing)', 'Beep, bleep', 'Busy signal', 
                 'Buzzer','Camera', 'Cash register', 'Chopping (food)', 'Clang', 'Clock', 'Coin (dropping)',
                 'Computer keyboard', 'Creak', 'Cupboard open or close', 'Cutlery, silverware', 'Dial tone',
                 'Ding', 'Ding-dong', 'Dishes, pots, and pans', 'Door', 'Doorbell', 'Drawer open or close',
                 'Filing (rasp)', 'Fire alarm', 'Frying (food)', 'Gears', 'Hammer', 'Keys jangling', 'Ping',
                 'Printer', 'Pulleys', 'Ratchet, pawl', 'Ringtone', 'Sanding', 'Sawing', 'Scissors',
                 'Sewing machine', 'Shuffling cards', 'Single-lens reflex camera', 'Sink (filling or washing)',
                 'Smoke detector, smoke alarm', 'Telephone', 'Telephone bell ringing', 'Telephone dialing, DTMF',
                 'Tick', 'Tick-tock', 'Toilet flush', 'Tools', 'Toothbrush', 'Typewriter', 'Typing', 'Writing',
                 'Zipper (clothing)'}


outdoor_sounds = {'Arrow', 'Bicycle', 'Boiling', 'Car alarm', 'Cluck', 'Crackle', 'Cricket',
                  'Fill (with liquid)', 'Gush', 'Ice cream truck, ice cream van', 'Knock', 'Patter', 'Pour',
                  'Power windows, electric windows', 'Pump (liquid)', 'Rumble', 'Rustle', 'Rustling leaves',
                  'Skateboard', 'Skidding', 'Slam', 'Sliding door', 'Spray', 'Squeak', 'Stir','Subway, metro, underground', 
                  'Tap', 'Thump, thud', 'Thunk', 'Tire squeal', 'Toot','Train wheels squealing', 'Trickle, dribble', 'Whir',
                  'Whoosh, swoosh, swish'}

urban_sounds = {'Air brake', 'Air horn, truck horn', 'Reversing beeps', 'Police car (siren)', 'Ambulance (siren)', 
                'Fire engine, fire truck (siren)', 'Traffic noise, roadway noise', 'Vehicle horn, car horn, honking', 
                'Railroad car, train wagon', 'Rail transport', 'Outside, urban or manmade', 'Civil defense siren',  
                'Siren', 'Fire','Splash, splatter', 'Bell', 'Church bell','Clickety-clack','Sonar','Steam'
}

enviromental_sounds = indoor_sounds | outdoor_sounds | urban_sounds | inhouse_sounds

nature_sounds = {
    'Wind noise (microphone)', 'Wind', 'Thunderstorm', 'Thunder', 'Water', 'Rain', 'Raindrop', 
    'Rain on surface', 'Stream', 'Waterfall', 'Ocean', 'Waves, surf', 
    'Outside, rural or natural'
}

animal_insects_sounds = {
    'Cattle, bovinae','Insect', 'Animal', 'Owl', 'Bird', 'Domestic animals, pets', 'Whale vocalization', 'Fowl', 
    'Wild animals', 'Bird vocalization, bird call, bird song', 'Hoot', 'Goose', 'Cat', 'Wail, moan', 
    'Dog', 'Bark', 'Howl', 'Meow', 'Livestock, farm animals, working animals', 'Clip-clop', 'Horse', 
    'Neigh, whinny', 'Moo', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Chicken, rooster', 
    'Crowing, cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Honk', 'Roaring cats (lions, tigers)', 
    'Roar', 'Chirp, tweet', 'Squawk', 'Pigeon, dove', 'Coo', 'Crow', 'Caw', 'Bird flight, flapping wings', 
    'Canidae, dogs, wolves', 'Rodents, rats, mice', 'Mouse', 'Mosquito', 'Fly, housefly', 'Bee, wasp, etc.', 
    'Frog', 'Croak', 'Snake', 'Rattle', 'Cowbell','Purr'
}

military_sounds = {
    'Explosion', 'Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire', 'Cap gun', 
    'Fireworks', 'Firecracker', 'Boom'
}

not_sure_sounds = {'Boat, Water vehicle','Bellow', 'Bang', 'Basketball bounce', 'Boing', 'Bouncing', 'Breaking', 'Clatter', 'Clicking',
                   'Crumpling, crinkling', 'Crunch', 'Crushing', 'Flap', 'Jingle, tinkle', 'Plop', 'Roll', 'Rub',
                   'Scrape', 'Scratch', 'Sizzle', 'Slap, smack', 'Smash, crash', 'Squeal', 'Tearing', 'Whack, thwack',
                   'Whip', 'Zing', 'Gurgling', 'Burst, pop', 'Eruption', 'Wood', 'Chop', 'Splinter', 'Crack', 'Glass', 
                   'Chink, clink', 'Shatter', 'Liquid', 'Slosh', 'Squish', 'Drip'}


# motors_sounds = daniel_motors_sounds.union(ours_motors_sounds)

ignore_sounds = {'Insect', 'Music', 'Outside, urban or manmade', 'Outside, rural or natural', 'Steam',  'Animal', 'Guitar', 'Clickety-clack', 'Wind noise (microphone)', 'Wind', 'Inside, small room', 'Civil defense siren', 'White noise', 'Siren', 'Musical instrument',  'Railroad car, train wagon', 'Rail transport', 'Speech', 'Drum', 'Drum kit', 'Bass drum', 'Percussion', 'Snare drum', 'Cymbal', 'Drum machine', 'Rimshot', 'Drum roll', 'Scary music', 'Thunderstorm', 'Thunder', 'Water', 'Rain', 'Raindrop', 'Rain on surface', 'Stream', 'Waterfall', 'Ocean', 'Waves, surf', 'Fire', 'Water tap, faucet', 'Whistle', 'Splash, splatter'}

ignore_warfare = {'Explosion', 'Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire', 'Cap gun', 'Fireworks', 'Firecracker', 'Boom'}

ignore_human = {'Male speech, man speaking', 'Female speech, woman speaking', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue', 'Speech synthesizer', 'Shout', 'Yell', 'Children shouting', 'Screaming'}

ignore_animals = {'Owl', 'Bird', 'Sonar', 'Domestic animals, pets', 'Whale vocalization', 'Fowl', 'Wild animals', 'Bird vocalization, bird call, bird song', 'Hoot', 'Goose', 'Cat', 'Wail, moan', 'Dog', 'Bark', 'Howl', 'Meow', 'Livestock, farm animals, working animals', 'Clip-clop', 'Horse', 'Neigh, whinny', 'Moo', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl', 'Chicken, rooster', 'Crowing, cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Honk', 'Roaring cats (lions, tigers)', 'Roar', 'Chirp, tweet', 'Squawk', 'Pigeon, dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Bird flight, flapping wings', 'Canidae, dogs, wolves','Rodents, rats, mice', 'Mouse',  'Mosquito', 'Fly, housefly', 'Bee, wasp, etc.', 'Frog', 'Croak', 'Snake', 'Rattle'}

ignore_traffic = {'Air brake', 'Air horn, truck horn', 'Reversing beeps', 'Police car (siren)', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Traffic noise, roadway noise', 'Vehicle horn, car horn, honking'}

ignore_not_sure = {'Bellow', 'Whispering', 'Laughter', 'Baby laughter', 'Giggle', 'Belly laugh', 'Chuckle, chortle', 'Crying, sobbing', 'Baby cry, infant cry', 'Sigh', 'Singing', 'Groan', 'Grunt', 'Breathing', 'Wheeze', 'Snoring', 'Walk, footsteps', 'Hubbub, speech noise, speech babble', 'Children playing', 'Growling', 'Purr', 'Cowbell', 'Bell', 'Church bell'}



ignore_all = ignore_sounds | ignore_warfare | ignore_human | ignore_animals | ignore_traffic | ignore_not_sure

other_sounds_no_ignor = [label for label in labels if label not in motors_sounds]
other_sounds_with_ignor = [label for label in other_sounds_no_ignor if label not in ignore_all]

# print('other_sounds_with_ignor =', other_sounds_with_ignor)
# print('len(other_sounds_with_ignor)=', len(other_sounds_with_ignor))
# print('len(motors_sounds)=', len(motors_sounds))
# #print('len(other_sounds_with_ignor) =', len(other_sounds_with_ignor))
# print('sum=',len(motors_sounds) + len(other_sounds_with_ignor))

# # Combine all dictionaries into a single list for easier processing
# all_sounds_combined = list(motors_sounds) + list(music_sounds) + list(silence_sounds) + list(noise_sounds) + list(military_sounds) + list(human_sounds) + list(enviromental_sounds) + list(nature_sounds) + list(animal_insects_sounds) + list(not_sure_sounds)
# print('all_sounds_combined list length=', len(all_sounds_combined))
# print('labels list length=', len(labels))

# # Convert lists to sets to efficiently find differences
# all_sounds_combined_set = set(all_sounds_combined)  # Use your actual all_sounds_combined list here
# labels_set = set(labels)  # Use your actual labels list here

# # Find sounds present in all_sounds_combined but not in labels
# sounds_not_in_labels = all_sounds_combined_set - labels_set

# # Find sounds present in labels but not in all_sounds_combined
# sounds_not_in_all_sounds_combined = labels_set - all_sounds_combined_set

# # Output the differences
# print("Sounds in 'all_sounds_combined' but not in 'labels':", sounds_not_in_labels)
# print("Sounds in 'labels' but not in 'all_sounds_combined':", sounds_not_in_all_sounds_combined)
# # Create a new dictionary to count occurrences
# sound_count_dict = {}

# # Go over each word in the combined list and count occurrences
# for sound in all_sounds_combined:
#     if sound in sound_count_dict:
#         sound_count_dict[sound] += 1
#     else:
#         sound_count_dict[sound] = 1

# # Find the sounds that appear more than once
# duplicated_sounds = {sound: count for sound, count in sound_count_dict.items() if count > 1}

# print(duplicated_sounds)
# print(sound_count_dict)
# print('')


