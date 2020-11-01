import React from 'react';
import {Link, useLocation} from "react-router-dom";
import {Button, Col, Form, FormControl, Nav, Navbar, NavDropdown, Row, Table} from "react-bootstrap";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}
let ResultTable = function (props) {
  return (
    <div>
      <Row className={"mt-5"}>
        <Col className={'col-10'}><h2>Результаты:</h2> </Col><Col className={'col-2'}><Button variant="outline-success">Экспорт в Excell</Button></Col>
      </Row>
      <Table striped bordered hover size="sm">
        <thead>
        <tr>
          <th>id</th>
          <th>Название</th>
          <th>Класс</th>
        </tr>
        </thead>
        <tbody>
        {props.result.map(function (val, i) {
          return <tr>
            <td><Link to={'/view/' + val.doc_id}>{val.doc_id}</Link></td>
            <td><Link to={'/view/' + val.doc_id}>{val.doc_src_name}</Link></td>
            <td><Link to={'/view/' + val.doc_id}>{val.doc_cls}</Link></td>
          </tr>
        })}
        </tbody>
      </Table>
    </div>
  )

}

export default ResultTable;