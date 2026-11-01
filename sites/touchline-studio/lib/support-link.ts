export function paypalContributionLink(base:string,amount:string,currency:string){
 if(!['USD','GBP','EUR'].includes(currency)||!/^\d+(\.\d{1,2})?$/.test(amount)||Number(amount)<=0||Number(amount)>10000)return '';
 try{const url=new URL(base);if(url.protocol!=='https:'||url.hostname!=='paypal.me')return '';const handle=url.pathname.split('/').filter(Boolean)[0];if(!handle)return '';return `https://paypal.me/${handle}/${Number(amount)}${currency}`}catch{return ''}
}
