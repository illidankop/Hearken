#author Ariel Zev the one and only

import numpy as np

BIN_MAGIC = b'\xff\xfe\xfd\xfc'
BIN_HEADER_SIZE = 80
PAYLOAD_SIZE = 80000
FRAME_SIZE = BIN_HEADER_SIZE + PAYLOAD_SIZE

def get_packet_lost_percentage(file_path):
    with open(file_path, 'rb') as bin_file:
        byte_data = bin_file.read()
        byte_array = np.frombuffer(byte_data, dtype=np.uint8)

        # check where magic word is
        magic_appears_bool = np.ones(byte_array.shape, dtype=bool)
        for i, let in enumerate(BIN_MAGIC):
            magic_appears_bool = np.bitwise_and(magic_appears_bool, np.roll(byte_array == let, -i))
        magic_appears_idx = np.argwhere(magic_appears_bool).squeeze()
        # first_hwtime = byte_array[magic_appears_idx[0] + 8:magic_appears_idx[0] + 12].view(np.uint32).squeeze()
        # last_hwtime = byte_array[magic_appears_idx[-1] + 8:magic_appears_idx[-1] + 12].view(np.uint32).squeeze()

        # check if the frame is full i.e. length of HEADER+PAYLOAD
        full_frames_idx = []
        for idx in magic_appears_idx:
            if idx + FRAME_SIZE in magic_appears_idx:
                full_frames_idx.append(idx)
        if magic_appears_idx[-1] + FRAME_SIZE == len(byte_array):  # add last frame if it has right size
            full_frames_idx.append(magic_appears_idx[-1])
        full_frames_idx = np.array(full_frames_idx)

    hwtimes = []
    for frame_idx in full_frames_idx:
        hwtimes.append(byte_array[frame_idx+8:frame_idx+12].view(np.uint32)[0])
    lost_packets = np.sum(np.diff(hwtimes) - 1)
    print(f'Number of lost packets: {lost_packets} in {len(hwtimes)} frames ({lost_packets/len(hwtimes)*100:.2f}%) for {file_path}')
    packet_lost_percentage = round(lost_packets/len(hwtimes)*100,2)
    return packet_lost_percentage