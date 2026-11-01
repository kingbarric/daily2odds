/** Final scores first; newest games and highest probabilities within each group. */
export function compareHistoryRows(a:{score:string|null;probability:number;match:{date:string}},b:{score:string|null;probability:number;match:{date:string}}){return Number(!a.score)-Number(!b.score)||b.match.date.localeCompare(a.match.date)||b.probability-a.probability}
