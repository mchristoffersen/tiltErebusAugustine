import obspy

def get_events():
    # AKDT
    events = ["2006-01-11T04:44:00",
              "2006-01-11T05:12:00",
              "2006-01-13T04:24:00",
              "2006-01-13T08:47:00",
              "2006-01-13T11:22:00",
              "2006-01-13T16:40:00",
              "2006-01-13T18:58:00",
              "2006-01-14T00:14:00",
              "2006-01-17T07:58:00",
              "2006-01-27T20:24:00",
              "2006-01-27T23:37:00",
              "2006-01-28T02:04:00",
              "2006-01-28T07:42:00",
              "2006-01-28T14:31:00"]

    # Events in UTC 
    return [obspy.UTCDateTime(event) + (60*60*9) for event in events]