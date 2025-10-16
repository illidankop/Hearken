import time
from threading import Thread


class MessageManager(Thread):
    MIN_TRACK_SIZE = 1

    def __init__(self, message_queue, tracker):
        super(MessageManager, self).__init__()
        self.message_queue = message_queue
        self.tracker = tracker
        self.process = True
        self.name = "Message Manager Thread"
        self.sent_messages = []

    def _get_messages(self):
        tracks = self.tracker.track_list
        messages = []  # out messages
        for tr in tracks:

            # only ones were not sent prior
            for it in tr.intercepts_calc:

                # advancing sent brier one by one to avoid multiprocess problem
                if len(tr) >= MessageManager.MIN_TRACK_SIZE and it.time > tr.last_sent_time:
                    tr.last_sent_time = it.time
                    messages.append(it.frame_result)
        return messages

    def _add_to_queue(self):
        messages = self._get_messages()
        self._clean_messages(messages)  # removed out of ICD attributes
        messages.sort(key=lambda x: x.updateTimeTagInSec)
        for msg in messages:
            self.message_queue.appendleft(msg)
            self.sent_messages.append(msg)

    @staticmethod
    def _clean_messages(messages):
        for msg in messages:
            del msg.sensor
            del msg.statistics

    def run(self):
        while self.process:
            self._add_to_queue()
            time.sleep(0.1)

    def stop(self):
        self.process = False

    def join(self, timeout: [float] = ...) -> None:
        self.process = False
        super(MessageManager, self).join()
