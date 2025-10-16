{
  "hearken_system_name": "Hearken",
  "output_base_path": "/outdir/SensorLeft/Hearken/",
  "routine_wave_file_location": "/Data/all_wave_files/",
  "log_level": "INFO",
  "log_backupCount": 20,
  "is_save_stream": "False",
  "save_results_history": "False",
  "mission": {
    "sample_rate": 32000,
    "sample_time_in_sec": 1,
    "is_training_mode": "False",
    "is_sniper_mode": "False",
    "is_urban_mode": "False",
    "is_use_filter_in_aoa_calculation": "True",
    "mode": {
      "proccessing": 1,
      "calibration": 0,
      "recordings": 0
    }
  },
  "mic_units": [
    {
      "COMMENT": "DON'T DELETE THIS UNIT - it's necessary for playback purpose",
      "unit_name": "playback_mics",
      "deployment": "Octi",
      "icd_version": 0.0,
      "unit_id": 1,
      "module_name": "micapi.playback_mics",
      "cls_name": "PlayBackMic",
      "playback_file_path": "F:/Injector/4PB/DS/failed/",
      "active": 0
    },
    {
      "unit_name": "syncope_mics",
      "deployment": "Octi",
      "unit_id": 1,
      "icd_version": 1.0,
      "module_name": "micapi.syncope",
      "cls_name": "SyncopeApi",
      "active": 0
    },
    {
      "unit_name": "caneri_mics",
      "deployment": "OtheloP_M",
      "printed_sensor_deployment": "OtheloP_P",
      "cnc_metal_sensor_deployment": "OtheloP_M",
      "icd_version": 1.0,
      "unit_id": 12,
      "module_name": "micapi.caneri",
      "cls_name": "CaneriApi",
      "active": 1
    },
    {
      "unit_name": "ia_mics",
      "deployment": "ia_mics",
      "icd_version": 1.0,
      "unit_id": 1,
      "module_name": "micapi.ia_mics",
      "cls_name": "AAMic",
      "active": 0
    },
    {
      "unit_name": "ba_mics",
      "deployment": "ba_mics",
      "icd_version": 1.0,
      "unit_id": 0,
      "module_name": "micapi.baringer",
      "cls_name": "BaMic",
      "active": 0
    },
    {
      "unit_name": "ba_mics",
      "deployment": "ba_mics",
      "icd_version": 1.0,
      "unit_id": 1,
      "module_name": "micapi.baringer",
      "cls_name": "BaMic",
      "active": 0
    }
  ],
  "proccesors": [
    {
      "unit_name": "gunshot_noisy",
      "module_name": "EAS.proccesers.eas_noisy_gunshot",
      "cls_name": "NoisyGunShotProcessor",
      "models": "",
      "active": 0
    },
    {
      "unit_name": "gunshot",
      "module_name": "EAS.proccesers.eas_gunshot",
      "cls_name": "GunShotProcessor",
      "models_files_path" : "/Data_base/models/",
      "models": "shot_model:gunshots32000_010822,shot_model_blank:hakshots32000_050223,bl_sw_model:BL_SW_32000_231022,sw_weapon_type_model:sw_weapon_types_230123",
      "active": 1
    },
    {
      "unit_name": "airborne",
      "module_name": "EAS.proccesers.eas_dg",
      "cls_name": "EasDgProcessor",
      "models": "",
      "active": 0
    },
    {
      "unit_name": "sp_airborne",
      "module_name": "EAS.proccesers.eas_sp_airborne",
      "cls_name": "Eas_SP_AirborneProcessor",
      "models": "airborne_model:dsmall_010522",
      "active": 0
    },
    {
      "unit_name": "airborneThreat",
      "module_name": "EAS.proccesers.eas_airthreat",
      "cls_name": "AirthreatProcessor",
      "models": "airborne_model:Cnn14_fft4096_64mels",
      "active": 0
    },
    {
      "unit_name": "atms",
      "module_name": "EAS.proccesers.eas_atm",
      "cls_name": "AtmProcessor",
      "models": "shot_model:atms32000,bl_sw_model:atms32000",
      "active": 0
    },
    {
      "unit_name": "atms",
      "module_name": "EAS.proccesers.eas_atm_ds",
      "cls_name": "AtmProcessor",
      "models": "shot_model:atms32000,bl_sw_model:atms32000",
      "active": 0
    },
    {
      "unit_name": "motors",
      "module_name": "EAS.proccesers.eas_motor",
      "cls_name": "MotorProcessor",
      "models": "shot_model:na,bl_sw_model:na",
      "active": 0
    }
  ],
  "clients": [
    {
      "name": "drone_gourd_icd",
      "icd_version": 1.0,
      "module_name": "EAS.icd",
      "cls_name": "DG_MsgDistributor",
      "ip": "128.78.100.52",
      "port": 7010,
      "is_server": 0,
      "active": 0
    },
    {
      "name": "gfp_icd",
      "icd_version": "2.0.5",
      "module_name": "EAS.icd.gfp_icd",
      "cls_name": "GFP_MsgDistributer",
      "ip": "127.0.0.1",
      "port": 18011,
      "is_server": 0,
      "active": 1
    },
    {
      "name": "msiss_icd",
      "icd_version": 2.0,
      "module_name": "EAS.icd.msiss_icd",
      "cls_name": "Hearken2Msiss_MsgDistributer",
      "ip": "127.0.0.1",
      "port": 3000,
      "is_server": 0,
      "active": 0
    },
    {
      "name": "gun_shot_icd",
      "icd_version": 1.0,
      "module_name": "EAS.icd",
      "cls_name": "GunShotMsgDistributor",
      "ip": "8.20.70.144",
      "port": 2000,
      "is_server": 0,
      "active": 0
    },
    {
      "name": "ui",
      "icd_version": 1.0,
      "module_name": "EAS.icd",
      "cls_name": "EasComm",
      "ip": "127.0.0.1",
      "port": 9090,
      "is_server": 1,
      "active": 0
    },
    {
      "name": "MAP_web_client",
      "icd_version": 1.0,
      "module_name": "EAS.icd.web_icd.web_handler",
      "cls_name": "AcousticframeResWebHandler",
      "ip": "10.15.110.25",
      "port": 3001,
      "is_server": 0,
      "active": 0
    }
  ],
  "constants": {
    "audio_files_length": 30,
    "keep_frames_history_sec": 120,
    "speed_of_sound": 331.5024,
    "temperature": 25,
    "humidity": 50
  },
  "ATM": {
    "probability_th": 0.8,
    "sectors_to_ignore_in_deg": [],
    "is_allways_valid_slope": "True",
    "max_event_power_threshold": 0.3,
    "min_aoas_4_check_slope": 10,
    "is_block_tta_check": "False",
    "aoa_event_power_th": 0.9,
    "srp_freq_range": [100,1000],
    "nfft": 1024,
    "event_conf_4_event": 80,
    "MIN_EVENTS_TIME_DIFF": 0.2,
    "MIN_CH_4_DETECT": 4,
    "AOA_WINDOW_SEC": 0.04
  },
  "algorithm": {
    "BPF_lowest_freq_th": 500
  },
  "calibration": {
    "port_name_lnx": "/dev/ttyUSB1",
    "port_name": "/dev/ttyUSB1",
    "use_gps": "False",
    "measured_azimuth": 163.71,
    "calibration_max_end_freq_HZ": 650,
    "calibration_freq_width_HZ": 200,
    "ses_lat": 29.96651,
    "ses_long": 34.815297,
    "ses_alt": 0,
    "tar_lat": 30.4075883,
    "tar_long": 34.942775,
    "tar_alt": 0,
    "offset": 0
  }
}