function FormatDate(timestamp) {
  let date = new Date(timestamp * 1000);
  return date.getDate() + "/" + date.getMonth() + "/" + date.getFullYear();
}

export {FormatDate};