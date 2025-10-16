class Icd_Serialzeable:
    
    def serialize(self, buf, offset):
        res = True
        try:
            res = self.to_bytes_array(buf, offset)            
        except Exception as ex:
            print(f'Failed to serialize - {ex}')
            self.logger.info(f'Failed to serialize - {ex}')
            res = False
        finally:            
            return res

    def deserialize(buf, offset):
        res = None
        try:
            res = self.from_bytes_array(buf, offset)
        except Exception as ex:
            self.logger.info(f'Failed to deserialize - {ex}')
            print(f'Failed to deserialize - {ex}')
            res = False
        finally:
            return res
    
    def to_bytes_array(self, buf, offset):
        return False

    def from_bytes_array(buf, offset):
        pass


# changed in version 3.2.9:
    # in serialize and deserialize return False value when failed