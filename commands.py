# Command IDs
CMD_READ_SWITCH        = 0x12

CMD_SET_DIR            = 0x20
CMD_TOGGLE_DIR         = 0x21

CMD_STEPPER_MOVE       = 0x30
CMD_HOME               = 0x31
CMD_MOVE_STATUS  	   = 0x33
CMD_GET_POSITION       = 0x37
CMD_SET_HOME_OFFSET    = 0x38
CMD_GET_HOME_OFFSET    = 0x39
STS_LIMIT_EVENT        = 0x34

CMD_AD9833_SINE_FREQ   = 0x40

CMD_START_SAMPLING     = 0x50
CMD_STOP_SAMPLING      = 0x51
CMD_SET_SAMPLING_FREQ  = 0x52
CMD_GET_SAMPLING_FREQ  = 0x53

CMD_PGA_SET            = 0x60

CMD_READ_TEMP          = 0x70


AMP_FFT = 0
AMP_RMS = 1

# Status
STS_ACK         = 0xFE
STS_ERR         = 0xFF
STS_LIMIT_HIT   = 0xEE

# ADC
USB_SAMPLES_PER_PACKET = 512   
SAMPLES_PER_FRAME = 4096         


# Sampling rate
FS = 20000

GAIN_LABEL_TO_CODE = {
    "1x": 0,
    "2x": 1,
    "5x": 2,
    "10x": 3,
    "20x": 4,
    "50x": 5,
    "100x": 6,
    "200x": 7,
}