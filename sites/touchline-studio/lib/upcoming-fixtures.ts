/** A fixture needs a verified future kickoff to appear in the live analyser. */
export function isUpcomingFixture(match:{kickoff?:string},now=Date.now()){
 const kickoff=Date.parse(match.kickoff||'');return Number.isFinite(kickoff)&&kickoff>now;
}
