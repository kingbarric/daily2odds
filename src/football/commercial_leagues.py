"""Commercial worker league registry; original private registry remains unchanged."""
from contextlib import contextmanager
from . import data

EXTRA={
 'D2':('Germany','2. Bundesliga',2),'SP2':('Spain','La Liga 2',2),
 'I2':('Italy','Serie B',2),'F2':('France','Ligue 2',2),
 'SC0':('Scotland','Premiership',1),'SC1':('Scotland','Championship',2),
 'SC2':('Scotland','League One',3),'B1':('Belgium','Pro League',1),
 'T1':('Turkey','Super Lig',1),'G1':('Greece','Super League',1)}
LEAGUES={**data.LEAGUES,**EXTRA}
SLUGS={'E0':'eng.1','E1':'eng.2','E2':'eng.3','E3':'eng.4','SP1':'esp.1','D1':'ger.1','I1':'ita.1','F1':'fra.1','N1':'ned.1','P1':'por.1',
 'D2':'ger.2','SP2':'esp.2','I2':'ita.2','F2':'fra.2','SC0':'sco.1','SC1':'sco.2','SC2':'sco.3','B1':'bel.1','T1':'tur.1','G1':'gre.1'}

@contextmanager
def league_scope():
    """Used only in the isolated nightly worker, never a request handler."""
    previous=dict(data.LEAGUES)
    data.LEAGUES.update(EXTRA)
    try:yield
    finally:data.LEAGUES.clear();data.LEAGUES.update(previous)

# Verified against the provider's per-competition metadata on 29 October 2026.
ESPN_IDS={'3927':'D2','3921':'SP2','3931':'I2','3926':'F2','735':'SC0','3940':'SC1','3941':'SC2','3901':'B1','3946':'T1','3955':'G1'}
TEAM_ALIASES={
 'St. Pauli':'St Pauli','Saint-Étienne':'St Etienne',"Queen's Park":'Queens Park',
 '1. FC Nürnberg':'Nurnberg','Dynamo Dresden':'Dresden','Karlsruher SC':'Karlsruhe','TSV Eintracht Braunschweig':'Braunschweig',
 'CD Sabadell':'Sabadell','FC Andorra':'Andorra','RC Celta Fortuna':'Celta B','Real Sociedad II':'Sociedad B','Real Oviedo':'Oviedo','Sporting Gijón':'Sp Gijon',
 'Clermont Foot':'Clermont','Dijon FCO':'Dijon','Pau':'Pau FC','Red Star FC 93':'Red Star','Rodez Aveyron':'Rodez','Stade Laval':'Laval',
 'Inverness Caledonian Thistle':'Inverness C','Raith Rovers':'Raith Rvs','Airdrieonians':'Airdrie Utd','Queen of the South':'Queen of Sth','Dunfermline Athletic':'Dunfermline','Greenock Morton':'Morton','Partick Thistle':'Partick','Ayr United':'Ayr',
 'KAA Gent':'Gent','KV Kortrijk':'Kortrijk','KV Mechelen':'Mechelen','Royal Charleroi SC':'Charleroi','Sint-Truidense':'St Truiden','Union St.-Gilloise':'St. Gilloise','Standard Liege':'Standard','Zulte-Waregem':'Waregem',
 'Amed SFK':'Amedspor','Caykur Rizespor':'Rizespor','Goztepe':'Goztep','Istanbul Basaksehir':'Buyuksehyr','Çorum FK':'Corum',
 'AEK Athens':'AEK','Asteras Tripoli':'Asteras Tripolis','Levadiakos':'Levadeiakos','Olympiacos':'Olympiakos'}
