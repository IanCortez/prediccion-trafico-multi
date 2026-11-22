from common import *
from datetime import date,timedelta
from dateutil.easter import easter

SOURCES={
'national_holiday_suspension':'https://diariodarepublica.pt/dr/detalhe/lei/23-2012-178501',
'restoration_2016':'https://files.diariodarepublica.pt/1s/2016/04/06400/0111201116.pdf',
'porto_june24':'https://bte.dgcp.mtsss.gov.pt/completos/2014/bte30_2014.pdf'
}
def special_dates(year):
    # Frozen study calendar: national days, Porto June 24 and Carnival as a local
    # activity indicator. It is not a determination of individual work obligations.
    days={date(year,m,d) for m,d in ((1,1),(4,25),(5,1),(6,10),(6,24),(8,15),(12,8),(12,25))}
    E=easter(year);days.update([E,E-timedelta(days=2),E-timedelta(days=47)])
    if not 2013<=year<=2015:days.update([E+timedelta(days=60),date(year,10,5),date(year,11,1),date(year,12,1)])
    return days

def calendar_columns(local_time):
    assert set(np.asarray(local_time.year).tolist())<={2013,2014},'Calendar certified for the frozen 2013-2014 Porto study only'
    base=date(1970,1,1).toordinal();holidays=np.array(sorted((d.toordinal()-base) for y in range(2012,2016) for d in special_dates(y)),np.int64)
    day=local_time.tz_localize(None).as_unit('s').asi8//86400;dow=np.asarray(local_time.dayofweek);special=np.isin(day,holidays);before=np.isin(day+1,holidays);after=np.isin(day-1,holidays)
    nxt=np.searchsorted(holidays,day);prev=np.searchsorted(holidays,day,side='right')-1
    until=np.minimum(holidays[nxt]-day,7)/7.;since=np.minimum(day-holidays[prev],7)/7.
    bridge=~special&(((dow==0)&before)|((dow==4)&after))
    return dict(calendar_special=special.astype(float),calendar_eve=before.astype(float),calendar_after=after.astype(float),calendar_bridge=bridge.astype(float),calendar_until_week=until,calendar_since_week=since,calendar_weekend_or_special=((dow>=5)|special).astype(float))
