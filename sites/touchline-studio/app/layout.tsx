import type {Metadata} from 'next';
import './globals.css';
export const metadata:Metadata={title:'Daily2Odds — AI-powered football predictions',description:'Free AI-powered football forecasts across 20 leagues. Nightly seven-day predictions using LightGBM, Dixon–Coles and Elo features. Optional community support.',icons:{icon:'/favicon.svg'}};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="en"><body>{children}</body></html>}
